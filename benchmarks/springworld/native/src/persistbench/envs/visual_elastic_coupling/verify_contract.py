"""Necessary causal indexing and privacy boundary risk checks."""
import json
from pathlib import Path
import numpy as np
from .schema import Episode,history_payload,query_packet
from .data import fixed_query_cases

def run():
    images=np.full((30,64,64),255,np.uint8); images[1:]=0
    actions=np.zeros((29,2),np.float32); actions[:,0]=np.linspace(0,.9,29)
    ep=Episode(images,actions,np.arange(30)*.05,np.zeros((30,8)),dict(episode_key="private_a",theta=[1,1,1],seed=2))
    h=history_payload(ep,0,23)
    assert h["observations"].shape==(24,2,64,64)
    assert np.all(h["observations"][1,1]==-1) and np.all(h["observations"][0,1]==0)
    assert np.array_equal(h["past_actions"][1:],actions[:23]) and not h["past_action_mask"][0]
    q0=query_packet(ep,3,0,4); q1=query_packet(ep,3,1,4)
    assert q0["observations"].shape[0]==1 and np.all(q0["observations"][0,1]==0)
    assert q1["observations"].shape[0]==2 and "past_actions" not in q1
    assert np.array_equal(q1["future_actions"],actions[3:7])
    for packet in (h,q0,q1):
        assert not set(packet).intersection({"private_metadata","theta","seed","system_id","condition","split","private_state"})
    try: query_packet(ep,28,1,3)
    except ValueError: pass
    else: raise AssertionError("incomplete future accepted")
    try: fixed_query_cases(ep,{"matched":ep},anchor=3,q=1,horizon=4)
    except ValueError: pass
    else: raise AssertionError("same-episode donor accepted")
    return dict(status="PASS",checks=["signed_difference_no_uint8_overflow","strict_24_frames_23_actions",
        "arrival_action_alignment","q0_no_hidden_predecessor","A_query_no_past_action",
        "complete_future_interval","private_field_projection","reject_same_episode_donor"],test_read=False)

if __name__=="__main__":
    result=run(); print(json.dumps(result))
    Path("reports/visual_elastic_coupling_v1/contract_checks.json").write_text(json.dumps(result,indent=2)+"\n")
