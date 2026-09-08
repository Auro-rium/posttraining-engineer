# Autonomous Post-Training Engineer Architecture
## AWS Agents for Humans Hackathon Submission

> Status: the repository currently runs a local Strands demonstration. The AWS
> services shown below are the intended deployment boundary, not live evidence.

### System Overview
```
┌─────────────────┐
│    USER / API   │  ← REST API Endpoints
└─────────┬───────┘
          │
┌─────────▼───────┐
│ ORCHESTRATOR    │  ← Workflow Control
└─────────┬───────┘
          │
     ┌────▼─────┐
     │  STATE   │  ← OptimizationRun (DynamoDB)
     └────┬─────┘
          │
┌─────────▼─────────────────────┐
│   EIGHT SPECIALIZED AGENTS    │  ← Strands Agents
│                               │
│  ┌─────────────┐  ┌─────────────┐  │
│  │Benchmark   │  │Failure      │  │
│  │ Agent      │  │Analyst     │  │
│  └─────────────┘  └─────────────┘  │
│  ┌─────────────┐  ┌─────────────┐  │
│  │Research    │  │Data         │  │
│  │ Agent      │  │Curator      │  │
│  └─────────────┘  └─────────────┘  │
│  ┌─────────────┐  ┌─────────────┐  │
│  │Training    │  │Training     │  │
│  │Designer    │  │Executor     │  │
│  └─────────────┘  └─────────────┘  │
│  ┌─────────────┐  ┌─────────────┐  │
│  │Eval        │  │Champion     │  │
│  │ Agent      │  │Manager      │  │
│  └─────────────┘  └─────────────┘  │
└─────────┬───────┬─────────────┘
          │       │
┌─────────▼───────┼─────────────────┐
│ AWS INFRASTRUCTURE              │
│                                 │
│  S3 Artifacts Store  ┌────────┐  │
│  ┌──────────────┐    │Executor│  │
│  │Trajectories  │    │(SageMaker)│ │
│  │FailureCls.   │    └────────┘  │
│  │Hypotheses    │    ┌────────┐  │
│  │Datasets      │    │  Eval  │  │
│  │Experiments   │    │Workers │  │
│  │Candidates    │    └────────┘  │
│  │Models        │    ┌────────┐  │
│  └──────────────┘    │CloudWatch│ │
│                      │(Monitoring)│
│  DynamoDB State ◄────┘           │
│  (OptimizationRun)               │
└─────────────────────────────────┘
```

### Agent Communication Flow

1. **User/API Layer** → **Orchestrator**
   - REST API calls to create runs, execute steps, check status
   - Orchestrator manages workflow progression

2. **Orchestrator** → **State Management**
   - Reads/writes OptimizationRun state (DynamoDB)
   - Tracks current phase, artifacts, performance metrics

3. **Orchestrator** → **Specialized Agents**
   - Routes requests to appropriate Strands agent based on phase
   - Each agent receives precise inputs and produces typed outputs

4. **Specialized Agents** → **AWS Services**
   - Benchmark/Eval Agents: AgentGym Environment Simulation
   - Training Executor: Amazon SageMaker (QLoRA training)
   - All Agents: S3 for artifact storage/retrieval
   - Optional: Amazon AgentCore for managed agent hosting
   - Optional: CloudWatch for observability

### Data Flow Example

```
User Request: POST /api/runs
        ↓
Orchestrator: Initialize OptimizationRun
        ↓
Orchestrator → Benchmark Agent: Run Initial Evaluation
        ↓
Benchmark Agent → S3: Store Trajectories
        ↓
Orchestrator ← Benchmark Agent: Trajectory References
        ↓
Orchestrator → State: Update trajectories[], baselinePerformance
        ↓
Orchestrator → Failure Analyst Agent: Analyze Failures
        ↓
Failure Analyst Agent → S3: Store Failure Clusters
        ↓
Orchestrator ← Failure Analyst Agent: Failure Analysis
        ↓
Orchestrator → State: Update failureClusters[]
        ↓
... (continues through all 8 agents) ...
        ↓
Orchestrator → Champion Manager: Make Promotion Decision
        ↓
Champion Manager → State: Update championCheckpoint if PROMOTE
        ↓
Orchestrator → State: Update currentPhase, championPerformance
        ↓
Response: Current Run Status
```

### Key Architecture Decisions

1. **Strands Agents Mandatory**: All eight specialized agents use AWS Strands Agents SDK
2. **AgentCore Optional**: Can be deployed to AgentCore Runtime for bonus points
3. **State-Driven**: Central OptimizationRun state in DynamoDB ensures deterministic behavior
4. **Artifact Storage**: S3 buckets for all intermediate and final artifacts
5. **Environment Simulation**: AgentGym service recovery environment for objective evaluation
6. **Deterministic Gates**: Champion Manager applies mathematical improvement/regression thresholds
7. **Modular Design**: Each agent has single responsibility and clear interfaces
8. **Observability**: Built-in logging, optional CloudWatch/AgentCore integration

### AWS Service Integration (Optional Enhancements)

```
For AgentCore Bonus Points:
┌─────────────────┐
│   API Gateway   │
└─────────┬───────┘
          │
┌─────────▼───────┐
│AgentCore Runtime│  ← Hosts Strands Agents
│(Managed Hosting)│
└─────────┬───────┘
          │
┌─────────▼─────────────────────┐
│   Strands Agent Workers       │
│   (Benchmark, Research, etc.) │
└─────────┬───────┬─────────────┘
          │       │
S3                                DynamoDB
 Artifacts                        State
```

### Security & Compliance

- **No Hardcoded Secrets**: All configuration through environment variables
- **Least Privilege**: IAM roles scoped to necessary actions
- **Data Protection**: S3 encryption, secure API communications
- **Audit Trail**: Complete workflow history in state and artifacts
