# AI ARMY FLYWHEEL - SINGLE SOURCE OF TRUTH
**Last Updated**: 2026-01-22
**Location**: This file is the AUTHORITATIVE reference for the flywheel architecture.

---

## MASTER ARCHITECTURE

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    AI ARMY DISNEY FLYWHEEL ECOSYSTEM                        │
│            "Every product strengthens the core, core strengthens all"       │
└─────────────────────────────────────────────────────────────────────────────┘

                              ┌─────────────────┐
                              │   AI ROUTER     │
                              │ (Central Brain) │
                              │  23x industry   │
                              └────────┬────────┘
                                       │
        ┌──────────────────────────────┼──────────────────────────────┐
        │                              │                              │
        ▼                              ▼                              ▼
┌───────────────┐            ┌─────────────────┐            ┌───────────────┐
│  MK COPILOT   │            │  TRAVEL AGENT   │            │  AIHANGOUT.AI │
│  (Business)   │◄──────────►│  (Preferences)  │◄──────────►│  (Community)  │
│  :3000        │            │  500M+ trained  │            │  (Coming)     │
└───────┬───────┘            └────────┬────────┘            └───────┬───────┘
        │                              │                              │
        └──────────────────────────────┼──────────────────────────────┘
                                       │
                                       ▼
                         ┌─────────────────────────┐
                         │     DATA FLYWHEEL       │
                         │  (Network Effects Hub)  │
                         └─────────────────────────┘
                                       │
        ┌──────────────────────────────┼──────────────────────────────┐
        │                              │                              │
        ▼                              ▼                              ▼
┌───────────────┐            ┌─────────────────┐            ┌───────────────┐
│  RTX 5090     │            │  DGX SPARK 1&2  │            │     AWS       │
│  (Curation)   │───────────►│   (Training)    │───────────►│   (Backup)    │
│  73,111/sec   │            │  120GB VRAM ea  │            │   S3/EC2      │
└───────────────┘            └─────────────────┘            └───────────────┘
```

---

## INFRASTRUCTURE LOCATIONS

### Primary Systems

| System | IP | Role | VRAM |
|--------|-----|------|------|
| **DGX Spark 1** | <spark1-ip> | Training, Inference | 120GB |
| **DGX Spark 2** | <spark2-ip> / <spark2-lan-ip> | Training, Inference | 120GB |
| **Windows RTX 5090** | Local | Data Curation (NeMo) | 32GB |
| **AWS** | Cloud | Backup, Overflow | Variable |

### Key Directories (on Spark-1: <spark1-ip>)

```
~/ai-business/
├── FLYWHEEL.md              ← THIS FILE (Single Source of Truth)
├── CLAUDE.md                ← Claude context file
├── STATE.md                 ← Current state documentation
│
├── flywheel/                ← FLYWHEEL PIPELINE CODE
│   ├── flywheel_config.yaml ← Master configuration
│   ├── data_qa.py           ← Data quality + WHY capture
│   ├── flywheel_audit.py    ← Audit/test framework
│   ├── flywheel_engine.py   ← Core engine
│   ├── feedback_loop.py     ← Feedback collection
│   ├── eval_gate.py         ← Evaluation gates
│   ├── auto_trainer.py      ← Training automation
│   └── FLYWHEEL_ANALYSIS_REPORT.md ← Latest audit
│
├── learning_loop/           ← LEARNINGS & IMPROVEMENTS
│   ├── LEARNINGS_DGX_SPARK.md ← Critical learnings
│   ├── adversarial_generator.py
│   └── issue_tracker.py
│
├── specialists/             ← PRODUCT COPILOTS
│   ├── mary_kay/            ← MK Copilot
│   │   ├── training/        ← Training data
│   │   └── logs/            ← Explainability logs
│   └── travel_agent/        ← Travel Agent AI
│
├── adapters/                ← TRAINED MODELS (55 adapters)
├── shared/                  ← CROSS-SYSTEM SHARED DATA
│   ├── chat/                ← AI-to-AI communication
│   └── learnings/           ← Backed up learnings
│
└── knowledge/               ← RAG KNOWLEDGE BASE
    └── hipporag/
```

---

## FLYWHEEL STAGES

### Stage 0: CONTRACT
**Config**: `flywheel/flywheel_config.yaml`
- Success metrics: 85% task success, 80% satisfaction
- Safety rules: 99% jailbreak resistance, 0% PII leak
- Eval suites: golden_v1, safety_v1, regression_v1

### Stage 1: DATA COLLECTION
**Code**: `flywheel/feedback_loop.py`, `flywheel/observability.py`
- Captures all interactions with full metadata
- Explainability logs at `specialists/*/logs/explainability.jsonl`
- Required fields: trace_id, timestamp, model_id, prompt, response, feedback

### Stage 1.5: DATA QA (WHY Capture)
**Code**: `flywheel/data_qa.py`
**Hardware**: RTX 5090 (x86_64) for NeMo Curator

**WHY Fields Captured**:
- `intent_classified` - What intent system detected
- `handler_used` - Which code path processed it
- `rag_used` / `rag_top_score` - Retrieval effectiveness
- `analysis_flags` - Quality issues detected
- `product_source` - Which product generated data

**Performance**: 73,111 samples/sec (RTX 5090), 90.8% filter pass rate

### Stage 2: TRAINING TRIGGER
**Code**: `flywheel/auto_trainer.py`
- Min 100 new examples before retrain
- 70% novelty threshold (not duplicates)
- 85% label quality required

### Stage 2.2: PREFERENCE TRAINING
**Code**: `flywheel/preference_trainer.py`
- Method: DPO (Direct Preference Optimization)
- Min 50 preference pairs
- Runs after SFT training

### Stage 2.5: EVAL GATE 1
**Code**: `flywheel/eval_gate.py`
- Golden pass rate: 95% required
- Safety pass rate: 100% required
- Max 5% regression allowed

### Stage 3: OPTIMIZATION
**Code**: `flywheel/model_optimizer.py`
- Methods: int8, fp8, int4 quantization
- 100 calibration samples

### Stage 4: DEPLOYMENT
**Code**: `flywheel/deployment_manager.py`
- Strategy: Canary (5% → 25% → 50% → 100%)
- Auto-rollback on 2% error increase

### Stage 5: MONITORING
**Code**: `flywheel/observability.py`
- Metrics: latency, throughput, error rate, quality
- Drift detection: input distribution, topic shift
- Continuous eval: 1% sample rate

### Stage 6: FEEDBACK
**Code**: `flywheel/feedback_loop.py`
- Explicit: thumbs up/down, corrections
- Implicit: task completion, abandonment
- Trust weighting: authenticated=1.0, anonymous=0.3

### Stage 7: LEARNING
**Code**: `flywheel/self_learning.py`, `flywheel/fix_loop.py`
- Auto-proposals: new training examples, eval slices, router rules
- Human approval required for production changes

---

## NETWORK EFFECTS (Disney Flywheel)

### Data Flow
```
MK Copilot ─────┐
                │
Travel Agent ───┼───► DATA QA ───► TRAINING ───► BETTER MODELS
                │      (WHY)         (QLoRA)          │
AIHangout.ai ───┘                                     │
       ▲                                              │
       └──────────────────────────────────────────────┘
                    (Models improve all products)
```

### Current Network Effect Status (Updated 2026-01-22)

| Product | Connected | Data Volume | Status |
|---------|-----------|-------------|--------|
| MK Copilot | YES | 1,357 examples | Active |
| MK Flagged | YES | 25 examples | Active |
| Travel Agent | YES | 1+ examples | **CONNECTED** |
| AIHangout.ai | YES | 1+ examples | **CONNECTED** |
| AI Router | YES | 1+ examples | **CONNECTED** |
| Hub Server | YES | 1+ examples | **CONNECTED** |
| Marketing | YES | 0 (ready) | Logging enabled |
| Sales | YES | 0 (ready) | Logging enabled |
| Support | YES | 0 (ready) | Logging enabled |
| HR | YES | 0 (ready) | Logging enabled |
| Legal | YES | 0 (ready) | Logging enabled |

**Network Effect Utilization**: 55% (6/11 products with data)
**Infrastructure Ready**: 100% (11/11 products can log)

---

## CRITICAL LEARNINGS

### 1. Training MUST use QLoRA (4-bit)
```python
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)
```

### 2. Inference MUST use vLLM (not Transformers)
- Transformers: ~3 tok/s (BAD)
- vLLM: ~50 tok/s (GOOD)

### 3. NeMo Curator requires x86_64
- DGX Spark (ARM64): Use fallback heuristics
- RTX 5090 (x86_64): Full NeMo Curator support

### 4. RTX 5090 requires PyTorch nightly
```bash
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128
```

### 5. Include adversarial training examples
- 35 adversarial examples proved effective
- Include language refusal, jailbreak resistance

---

## COMMANDS REFERENCE

### Run Flywheel Audit
```bash
cd ~/ai-business && python flywheel/flywheel_audit.py --full --save
```

### Check Training Status
```bash
tail -f /tmp/mk_training_cuda.log
```

### Start MK Copilot Server
```bash
cd ~/ai-business/specialists/mary_kay && ./start_server.sh
```

### Sync to Spark-2
```bash
rsync -avz ~/ai-business/shared/ <user>@<spark2-lan-ip>:~/ai-business/shared/
```

### Backup to AWS
```bash
aws s3 sync ~/ai-business/learning_loop/ s3://ai-army-backup/learnings/
```

---

## STAGE 8: SELF-AUTOMATION (Disney Flywheel)

### Self-Learning
**Code**: `flywheel/self_learning.py`
- Learns from pipeline runs (failures, successes)
- Learns from user feedback (thumbs up/down)
- Learns from incidents (root cause → fix)
- Knowledge gap detection

### Self-Healing
**Code**: `flywheel/self_healing.py`
- Continuous health monitoring
- Auto-restart unhealthy services
- Auto-scaling on load
- Auto-rollback on failures
- Circuit breaking for dependencies

### Self-Improvement
**Code**: `flywheel/autonomous_improver.py`
- Generates fixes for detected issues
- Validates fixes before applying
- Tracks fix effectiveness
- Categories: load_performance, input_validation, configuration

### Autonomous Runner
**Code**: `flywheel/autonomous_runner.py`
- Master daemon connecting all self-* systems
- Scheduled intervals for health/learning/improvement
- Modes: full_auto, supervised, monitoring, disabled

### Commands
```bash
# Run single cycle
python flywheel/autonomous_runner.py --once

# Run as daemon (supervised mode)
./start_autonomous.sh --daemon

# Run as daemon (full auto mode)
./start_autonomous.sh --full-auto

# Check status
./start_autonomous.sh --status

# Stop daemon
./start_autonomous.sh --stop
```

---

## AUDIT SCHEDULE

| Audit Type | Frequency | Command |
|------------|-----------|---------|
| Autonomous Cycle | Every 5 min | Cron: `autonomous_runner.py --once` |
| Flywheel Health | Hourly | Cron: `flywheel_audit.py --full` |
| Data Quality | Weekly | `python flywheel/data_qa.py --analyze` |
| Network Effects | Weekly | Check product connection status |
| Full Backup | Weekly | Sync to AWS S3 |

---

**DO NOT CREATE DUPLICATE FLYWHEEL DOCUMENTATION**
**THIS FILE IS THE SINGLE SOURCE OF TRUTH**
**All updates go here: ~/ai-business/FLYWHEEL.md**
