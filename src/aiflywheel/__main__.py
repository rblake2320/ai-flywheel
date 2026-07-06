"""
Runnable demo: `python -m aiflywheel demo`

Spins a real multi-tenant flywheel with the full pipeline (curator → floor →
learner → accelerometer), three vertical tenants, and prints the wheel turning:
per-batch acceleration, hub coverage, and per-tenant lift. No GPU, no network.
"""
from __future__ import annotations

import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from aiflywheel.contract.sdk import FlywheelClient
from aiflywheel.core.learner import FewShotLearner
from aiflywheel.curation.curator import default_curator
from aiflywheel.engine import FlywheelEngine
from aiflywheel.metrics.promotion import PromotionGate
from aiflywheel.tenancy.tenant import Tenant


def demo() -> int:
    eng = FlywheelEngine(
        batch_size=20,
        learner=FewShotLearner(bank_size=6),
        curator=default_curator(min_reward=0.3),
        promotion=PromotionGate(),          # close the loop: promote/rollback
    )
    tenants = [("mk-copilot", "retail"), ("realty-bot", "real_estate"),
               ("car-sales", "automotive")]
    clients = []
    for tid, dom in tenants:
        eng.add_tenant(Tenant(tenant_id=tid, domain=dom))
        clients.append((FlywheelClient(eng, tid), dom))

    print("ai-flywheel demo — 3 tenants, full pipeline\n" + "=" * 44)
    for n in range(180):
        client, dom = clients[n % len(clients)]
        client.report(
            input_text=f"{dom} question {n}",
            output_text=f"good {dom} answer {n}",
            reward=0.6 + 0.4 * ((n % 5) / 4),          # spread of quality
            domain=dom,
            cross_learning=f"{dom} pattern {n % 7} improves outcomes",
            provenance="real" if n % 4 else "synthetic",
        )
    eng.flush()

    h = eng.health()
    acc = h["acceleration"]
    print(f"\nmodel quality : {h['model_quality']} (peak {acc['peak_quality']})")
    print(f"acceleration  : climbed then {acc['status'].lower()} "
          f"(did_accelerate={acc['did_accelerate']}, {acc['batches']} batches)")
    print(f"loop closure  : {h['promotions']} promotions, {h['rollbacks']} rollbacks "
          f"(self-corrects — a regressing train is reverted)")
    sm = eng.self_model()
    conf = sm.confidence()
    print(f"self-model    : confidence {conf['score']} ({conf['verdict']}), "
          f"self-check consistent={sm.self_check()['consistent']}")
    gaps = sm.known_gaps()
    print(f"knows it lacks: {gaps[0]}")
    print(f"hub           : {h['hub']['total_learnings']} learnings, "
          f"networked={h['hub']['is_networked']}, "
          f"domains={h['hub']['domains']}")
    print("\nper-tenant lift (network effect — >1.0 means net winner):")
    net_winner = True
    for tid, _ in tenants:
        eng.pull(tid)
        lift = eng.lift.lift(tid)
        print(f"  {tid:12} gained={lift['gained']:3} contributed={lift['contributed']:3} "
              f"lift={lift['lift_ratio']}x")
        net_winner = net_winner and lift["lift_ratio"] > 1.0
    # the flywheel is "turning" when it is networked AND every tenant is a net
    # winner (gets more out than it puts in). Acceleration status is reported
    # separately and honestly.
    print(f"\nnetworked across {len(h['hub']['domains'])} domains: {h['hub']['is_networked']}")
    print(f"every tenant a net winner (lift > 1.0): {net_winner}")
    print("=> FLYWHEEL IS TURNING" if (h["hub"]["is_networked"] and net_winner)
          else "=> not yet turning")
    return 0


def organism_demo() -> int:
    """Run the whole flywheel as one connected organism — everything feeds
    everything. Shows the connections firing and the novel directions it seeks."""
    from aiflywheel.core.learner import FewShotLearner
    from aiflywheel.metrics.promotion import PromotionGate
    from aiflywheel.organism import Organism

    eng = FlywheelEngine(batch_size=15, learner=FewShotLearner(),
                         promotion=PromotionGate())
    for tid, dom in [("mk-copilot", "retail"), ("realty-bot", "real_estate"),
                     ("car-sales", "automotive")]:
        eng.add_tenant(Tenant(tid, domain=dom))
    clients = [(FlywheelClient(eng, t), d) for t, d in
               [("mk-copilot", "retail"), ("realty-bot", "real_estate"),
                ("car-sales", "automotive")]]
    for n in range(90):
        c, dom = clients[n % 3]
        c.report(input_text=f"{dom} question {n}", output_text=f"good {dom} answer {n}",
                 reward=0.7 + 0.3 * ((n % 4) / 3), domain=dom,
                 cross_learning=f"{dom} pattern {n % 6} improves outcomes")

    org = Organism(engine=eng)
    print("ai-flywheel ORGANISM — one connected complex run\n" + "=" * 48)
    report = org.run_cycle()
    print("\nconnections that fired this cycle:")
    for c in report.connections_fired:
        print(f"  • {c}")
    print(f"\nself-awareness : confidence {report.confidence} ({report.verdict}), "
          f"consistent={report.self_consistent}")
    print(f"explore/exploit: budget {report.explore_budget} novel directions "
          f"(low confidence → explore more)")
    print("\nnovel directions it chose to explore (self-directed, nobody set these):")
    for f in report.frontiers:
        print(f"  [{f.novelty}] {f.source:9} {f.description[:60]}")
    exps = org.frontiers_as_experiments(report.frontiers)
    print(f"\n{len(exps)} became SYNTHETIC experiments — the collapse floor bounds "
          "them automatically (curiosity that can't collapse the model)")
    print("\n=> THE ORGANISM IS RUNNING — it learns, corrects, reflects, knows "
          "itself, AND seeks the unknown")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "demo":
        return demo()
    if argv and argv[0] == "organism":
        return organism_demo()
    print("usage: aiflywheel [demo|organism]")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
