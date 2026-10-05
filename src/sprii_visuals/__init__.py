"""One display API for simulator states and original observation frames.

This module NEVER advances a simulator, samples a system, or modifies its
observation encoder. Physics / observation generation stay in native backends.
Use draw_scene for vector export and render_rgb for human-facing live displays.
"""
from pathlib import Path
import json
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch

P = json.loads((Path(__file__).with_name('palette.json')).read_text())
ENVIRONMENTS = ('spring', 'dclean', 'poke', 'swimmer', 'cophy', 'overcooked', 'baxter', 'rh20t', 'nod1d', 'nod2d')

def _field(ax, bounds, border=True):
    ax.set_facecolor(P['panel']);ax.set_xlim(*bounds[:2]);ax.set_ylim(*bounds[2:])
    ax.set_aspect('equal');ax.set_xticks([]);ax.set_yticks([])
    for sp in ax.spines.values():sp.set_visible(border);sp.set_color(P['grid']);sp.set_linewidth(.6)

def spring_line(ax, p1, p2):
    p1,p2=np.array(p1,float),np.array(p2,float);d=p2-p1
    u=d/max(np.linalg.norm(d),1e-9);v=np.array([-u[1],u[0]])
    # Fixed amplitude encodes no stiffness estimate.
    pts=[p1]+[p1+t*d+v*.025*(-1)**j for j,t in enumerate(np.linspace(.12,.88,13))]+[p2]
    pts=np.array(pts);ax.plot(*pts.T,color=P['spring'],lw=1.15,zorder=2,solid_capstyle='round')

def _force(ax, origin, action, scale):
    action=np.asarray(action,float)
    if np.linalg.norm(action)>1e-6:
        ax.add_patch(FancyArrowPatch(origin,np.asarray(origin)+scale*action,arrowstyle='-|>',mutation_scale=6,lw=.9,color=P['force'],zorder=6))

def draw_scene(ax, environment, *, state=None, action=None, contact=False, frame=None,
               bounds=None, border=True, trace=None, tactile_limit=None, field_limit=None):
    """Render state or native observation on an existing Matplotlib Axes.

    State contracts: spring [p1x,p1y,p2x,p2y,...], poke
    [fx,fy,fvx,fvy,ox,oy,ovx,ovy], dclean [x,y,vx,vy].
    Poke radii are the native 0.06 and 0.09 m (not enlarged per figure).
    All positions share a fixed environment viewport unless explicitly supplied.
    Recorded / third-party frames keep native aspect and pixels. Swimmer uses
    the same documented display LUT as the appendix; the raw frame is preserved.
    """
    if environment not in ENVIRONMENTS:raise ValueError(environment)
    if environment in ('spring','poke','dclean'):
        x=np.asarray(state,float)
        expected={'spring':(4,8),'poke':(8,),'dclean':(4,)}[environment]
        if x.ndim!=1 or len(x) not in expected or not np.isfinite(x).all():raise ValueError('Invalid state shape or nonfinite state')
        default={'spring':(-.37,.37,-.28,.28),'poke':(-1,1,-1,1),'dclean':(-1.5,1.5,-1.5,1.5)}
        _field(ax,bounds or default[environment],border)
        if trace is not None:
            z=np.asarray(trace,float)
            for inds,c in ([(0,P['body1']),(4,P['body2'])] if environment=='poke' else [(0,P['body1'])]):
                ax.plot(z[:,inds],z[:,inds+1],color=c,alpha=.42,lw=.65,zorder=1)
        if environment=='spring':
            p1,p2=x[:2],x[2:4];spring_line(ax,p1,p2)
            for q,c in [(p1,P['body1']),(p2,P['body2'])]:ax.add_patch(Circle(q,.055,fc=c,ec='white',lw=.5,zorder=3))
            if action is not None:_force(ax,p1,action,.18)
        elif environment=='poke':
            for q,r,c in [(x[:2],.06,P['body1']),(x[4:6],.09,P['body2'])]:ax.add_patch(Circle(q,r,fc=c,ec='white',lw=.45,zorder=3))
            if action is not None:_force(ax,x[:2],action,.22)
            if contact:ax.add_patch(Circle(x[4:6],.12,fill=False,ec=P['force'],lw=.85,zorder=4))
        else:
            ax.add_patch(Circle(x[:2],.14,fc=P['body1'],ec='white',lw=.6,zorder=3))
            ax.add_patch(Circle(x[:2],.20,fill=False,ec=P['spring'],lw=.7,ls='--'))
            if action is not None:_force(ax,x[:2],action,.14)
    elif environment in ('nod1d','nod2d'):
        z=np.asarray(state,float)
        if z.ndim!=2 or not np.isfinite(z).all():raise ValueError('PDE field must be a finite 2D array')
        from matplotlib.colors import LinearSegmentedColormap
        cm=LinearSegmentedColormap.from_list('pde',[P['heat_low'],P['canvas'],P['heat_high']])
        lim=field_limit or float(np.abs(z).max()) or 1
        if not np.isfinite(lim) or lim<=0:raise ValueError('Field limit must be positive and finite')
        ax.imshow(z,origin='lower',aspect='auto' if environment=='nod1d' else 'equal',extent=bounds or (0,1,0,1),cmap=cm,vmin=-lim,vmax=lim,interpolation='nearest')
        ax.set_xticks([]);ax.set_yticks([])
        for sp in ax.spines.values():sp.set_visible(border);sp.set_color(P['grid']);sp.set_linewidth(.6)
    elif environment=='baxter':
        z=np.asarray(state,float)
        if z.ndim!=2 or z.shape[0]!=16:raise ValueError('Baxter expects 16 sensors x time')
        lim=tactile_limit or float(np.abs(z).max()) or 1
        from matplotlib.colors import LinearSegmentedColormap
        cm=LinearSegmentedColormap.from_list('tactile',[P['body2'],'#FFFFFF',P['body1']])
        ax.imshow(z,origin='lower',aspect='auto',cmap=cm,vmin=-lim,vmax=lim,interpolation='nearest');ax.set_xticks([]);ax.set_yticks([])
        for sp in ax.spines.values():sp.set_visible(False)
    else:
        arr=np.array(frame,copy=True)
        if arr.ndim!=3 or arr.shape[2] not in (3,4):raise ValueError('Native frame must be H x W x RGB(A)')
        if environment=='swimmer':
            arr=arr[...,:3].astype(float);arr=arr/255 if arr.max()>1 else arr
            m=arr.max(2);out=np.empty_like(arr);out[:]=[.98,.97,.95];mask=m>.012
            lum=np.clip(m[mask]*2.4,0,1)[:,None]
            out[mask]=np.array([.25,.08,.12])*(1-lum)+np.array([.63,.34,.40])*lum;arr=out
        ax.imshow(arr,interpolation='nearest');ax.set_aspect('equal');ax.axis('off')
    return ax

def render_rgb(environment, *, width=320, height=240, **kwargs):
    """Pure human-facing RGB display; input arrays are never mutated."""
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    fig=Figure(figsize=(width/100,height/100),dpi=100,facecolor='white');canvas=FigureCanvasAgg(fig)
    draw_scene(fig.add_axes([0,0,1,1]),environment,**kwargs);canvas.draw()
    return np.asarray(canvas.buffer_rgba())[...,:3].copy()
