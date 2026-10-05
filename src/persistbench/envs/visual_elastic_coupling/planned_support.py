"""Bind preallocated evaluation cases to exact physical episode slots.

No success-based resampling, modulo reuse, or random re-selection is permitted.
The returned history packets remain the existing public codec. Planning jobs
does not run the generator or grant access to sealed test assets.
"""
import hashlib
from .evaluation_blocks import validate_plan
from .research_bank import episode_seed
from .prediction_assay import history_case


def planned_jobs(plan,systems,namespace):
    validate_plan(plan);systems=list(systems)
    if [s['system_key'] for s in systems]!=plan['system_generation_order']:raise ValueError('population generation order differs from plan')
    if any(s['stratum']!=plan['stratum'] for s in systems):raise ValueError('mixed planned physical populations')
    if not namespace:raise ValueError('fresh explicit episode namespace required')
    by_key={s['system_key']:s for s in systems};slots={key:set() for key in by_key}
    for case in plan['cases']:
        for role,indices in case['episode_slots'].items():
            kind={'control_cold':'cold','interference_moving':'moving'}.get(role,role)
            slots[case['system_key']].update((kind,index) for index in indices)
    jobs=[]
    for system in systems:
        for kind,index in sorted(slots[system['system_key']]):
            seed=episode_seed(namespace,system['split'],system['system_key'],kind,index)
            key=hashlib.sha256(f'{namespace}:{seed}:{kind}'.encode()).hexdigest()[:32]
            jobs.append(dict(episode_key=key,seed=seed,kind=kind,replicate=index,system=system))
    if len({j['episode_key'] for j in jobs})!=len(jobs):raise ValueError('planned episode identity collision')
    return jobs


class PlannedSupport:
    def __init__(self,bank,plan,*,split):
        validate_plan(plan)
        if split not in ('validation','test'):raise ValueError('planned evaluation has no training mode')
        self.bank=bank;self.plan=plan;self.split=split;self.cases={r['case_id']:r for r in plan['cases']};self.slots={}
        for row in bank.rows.values():
            if row['split']!=split or row['system_key'] not in plan['system_generation_order']:continue
            if row['stratum']!=plan['stratum']:raise ValueError('bank stratum differs from plan')
            slot=(row['system_key'],row['kind'],row['replicate'])
            if slot in self.slots:raise ValueError('duplicate bank episode slot')
            self.slots[slot]=row
        self.required_slots={}
        for case in plan['cases']:
            for role,indices in case['episode_slots'].items():
                kind={'control_cold':'cold','interference_moving':'moving'}.get(role,role)
                self.required_slots.setdefault((case['system_key'],kind),set()).update(indices)
        for (system,kind),indices in self.required_slots.items():
            for index in indices:
                if (system,kind,index) not in self.slots:raise ValueError('planned episode is missing; no fallback sampling')
        episode_keys=[self.slots[(s,k,i)]['episode_key'] for (s,k),indices in self.required_slots.items() for i in indices]
        if len(set(episode_keys))!=len(episode_keys):raise ValueError('independent planned slots share an episode')

    def row(self,system,kind,index,frames):
        row=self.slots[(system,kind,index)]
        if row['raw_frames']<frames:raise ValueError('planned observed support unavailable; retain failure instead of resampling')
        return row

    def prepare(self,case_id,*,query_kind='cold',query_budget=0,horizon=16,conditions=('null','matched96','wrong96'),factor_bank=None):
        case=self.cases[case_id];system=case['system_key'];wrong=case['wrong_system_key'];slots=case['episode_slots']
        conditions=tuple(conditions);sources={('matched96' if c=='matched96_after_query_interference' else c) for c in conditions}
        # Resolve only the support actually consumed by this registered assay.
        # An unrelated episode or a later, unscored horizon cannot filter it.
        forced=(system,'forced',slots['forced'][0]);wrong_slot=(wrong,'forced',slots['forced'][0])
        mass=[(system,'mass',i) for i in slots['mass']];free=[(system,'free',i) for i in slots['free']]
        members={'null':[],'matched24':[(forced,24)],'matched48':[(forced,48)],'matched96':[(forced,96)],'wrong96':[(wrong_slot,96)],
            'M':[(mass[0],48)],'F':[(free[0],48)],'MM':[(r,48) for r in mass],'FF':[(r,48) for r in free],
            'MF':[(mass[0],48),(free[0],48)],'repeat_M':[(mass[0],48),(mass[0],48)],'repeat_F':[(free[0],48),(free[0],48)]}
        histories={};budgets={}
        for condition,selection in members.items():
            if condition not in sources:continue
            selected=[(self.row(*slot,n),n) for slot,n in selection]
            histories[condition]=[history_case(self.bank,row,n) for row,n in selected]
            unique={r['episode_key']:n for r,n in selected}
            budgets[condition]=dict(donor_episode_keys=[r['episode_key'] for r,n in selected],processed_frames=sum(n for _,n in selected),
                unique_frames=sum(unique.values()),unique_transitions=sum(n-1 for n in unique.values()),episodes=len(selected),
                effort_n2_s=sum(float((self.bank.visible(r['episode_key'])[1][:n-1]**2).sum()*.05) for r,n in selected))
        if any(c.startswith('factor_') for c in sources) and factor_bank is not None:
            original,extra,metadata=factor_bank.histories((self.split,system),case['replicate'],self.bank.resolution)
            if original!=self.row(*forced,96)['episode_key']:raise ValueError('factor donor source differs from the planned original episode')
            histories.update(extra);budgets.update(metadata)
        if not sources.issubset(histories):raise ValueError('planned condition has no registered history')
        if query_kind not in ('cold','moving') or horizon not in (1,4,16,32):raise ValueError('unregistered planned query')
        anchor=1 if query_kind=='cold' else 95
        if query_budget not in ((0,1) if query_kind=='cold' else (0,1,3,7,15,31,63,95)):raise ValueError('unregistered planned query budget')
        query=self.row(system,query_kind,slots[query_kind][0],anchor+horizon+1)
        if query['anchor']!=anchor or (query_kind=='cold' and query['template']!='pulse_050'):raise ValueError('query slot does not match the registered task')
        interference=[self.row(wrong,'moving',i,112) for i in slots['interference_moving']] if 'matched96_after_query_interference' in conditions else []
        donor_ids={key for b in budgets.values() for key in b['donor_episode_keys']}
        if query['episode_key'] in donor_ids:raise ValueError('query and donor supports overlap')
        return histories,budgets,{query_kind:query},interference
