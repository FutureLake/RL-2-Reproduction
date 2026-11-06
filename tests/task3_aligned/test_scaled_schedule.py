import math
import pytest
from rl2_reproduction.task3_scaled.config import learning_rate_at,SCHEDULE

def test_schedule_boundaries_and_monotonic_segments():
    assert learning_rate_at(5000498)==3e-4
    assert learning_rate_at(5250498)==pytest.approx(4.5e-4)
    assert learning_rate_at(5500498)==6e-4
    assert learning_rate_at(10000000)==6e-4
    assert learning_rate_at(15000000)==pytest.approx(3.15e-4)
    assert learning_rate_at(20000000)==3e-5
    assert learning_rate_at(20000499)==3e-5
    up=[learning_rate_at(x) for x in range(5000498,5500499,50000)]
    down=[learning_rate_at(x) for x in range(10000000,20000500,50000)]
    assert all(a<=b for a,b in zip(up,up[1:]))
    assert all(a>=b for a,b in zip(down,down[1:]))

def test_schedule_resume_is_action_based_not_process_or_update_based():
    assert [learning_rate_at(x) for x in (5000498,5510000,9980000,15005500)]==[
        learning_rate_at(x,dict(SCHEDULE)) for x in (5000498,5510000,9980000,15005500)]
