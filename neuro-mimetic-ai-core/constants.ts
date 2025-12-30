import { BrainNodeData, BrainRegionType, PipelineStage, PipelineStepResult } from './types';

// =============================================================================
// COLOR PALETTE (Cyberpunk/Neural Theme)
// =============================================================================
export const COLORS: Record<string, string> = {
  // Regions
  THALAMUS: '#ff6b6b',        // Red - Input Gate
  SENSORY_CORTEX: '#ffd93d',  // Yellow - Perception
  PFC: '#00f3ff',             // Cyan - Executive
  DLPFC: '#6bcfff',           // Light Blue - Working Memory
  APFC: '#4ecdc4',            // Teal - Metacognition
  BASAL_GANGLIA: '#ff9f43',   // Orange - Decision Gate
  AMYGDALA: '#ee5a24',        // Deep Orange - Safety
  MOTOR_CORTEX: '#00ff88',    // Green - Action
  HIPPOCAMPUS: '#a55eea',     // Purple - Memory
  
  // UI
  BACKGROUND: '#050508',
  SUCCESS: '#00ff88',
  WARNING: '#ffd93d',
  ERROR: '#ff6b6b',
  INFO: '#00f3ff',
};

// =============================================================================
// BRAIN NODES - Anatomically Positioned for Pipeline Visualization
// =============================================================================
export const BRAIN_NODES: BrainNodeData[] = [
  // --- INPUT LAYER (Bottom) ---
  {
    id: 'thalamus',
    type: BrainRegionType.THALAMUS,
    position: [0, -1.5, 0],
    color: COLORS.THALAMUS,
    label: 'THALAMUS',
    description: 'Sensory Relay Station',
    pipelineStage: PipelineStage.SENSORY,
  },
  {
    id: 'sensory',
    type: BrainRegionType.SENSORY_CORTEX,
    position: [0, -0.8, -1.5],
    color: COLORS.SENSORY_CORTEX,
    label: 'SENSORY',
    description: 'Input Parser',
    pipelineStage: PipelineStage.SENSORY,
  },
  
  // --- PROCESSING LAYER (Middle-Front) ---
  {
    id: 'dlpfc',
    type: BrainRegionType.DLPFC,
    position: [-1, 1.2, 1.8],
    color: COLORS.DLPFC,
    label: 'dlPFC',
    description: 'Working Memory',
    pipelineStage: PipelineStage.WORKING_MEMORY,
  },
  {
    id: 'apfc',
    type: BrainRegionType.APFC,
    position: [1, 1.2, 1.8],
    color: COLORS.APFC,
    label: 'aPFC',
    description: 'Metacognition',
    pipelineStage: PipelineStage.METACOGNITION,
  },
  {
    id: 'pfc',
    type: BrainRegionType.PFC,
    position: [0, 1.5, 2.2],
    color: COLORS.PFC,
    label: 'PFC',
    description: 'Executive Planner',
    pipelineStage: PipelineStage.PLANNING,
  },
  
  // --- GATING LAYER (Center) ---
  {
    id: 'basal-ganglia',
    type: BrainRegionType.BASAL_GANGLIA,
    position: [0, 0, 0],
    color: COLORS.BASAL_GANGLIA,
    label: 'BASAL GANGLIA',
    description: 'Inhibition Gate',
    pipelineStage: PipelineStage.INHIBITION,
  },
  {
    id: 'amygdala',
    type: BrainRegionType.AMYGDALA,
    position: [1.2, -0.3, 0.5],
    color: COLORS.AMYGDALA,
    label: 'AMYGDALA',
    description: 'Safety Filter',
    pipelineStage: PipelineStage.INHIBITION,
  },
  
  // --- OUTPUT LAYER (Top) ---
  {
    id: 'motor',
    type: BrainRegionType.MOTOR_CORTEX,
    position: [0, 2.2, 0],
    color: COLORS.MOTOR_CORTEX,
    label: 'MOTOR',
    description: 'Action Executor',
    pipelineStage: PipelineStage.EXECUTION,
  },
  
  // --- MEMORY (Side) ---
  {
    id: 'hippocampus',
    type: BrainRegionType.HIPPOCAMPUS,
    position: [-1.5, 0, 0],
    color: COLORS.HIPPOCAMPUS,
    label: 'HIPPOCAMPUS',
    description: 'Memory/Context',
    pipelineStage: PipelineStage.REWARD,
  },
];

// =============================================================================
// NEURAL PATHWAYS - Connections between regions (for signal animation)
// =============================================================================
export const NEURAL_PATHWAYS: Array<{ from: string; to: string; label?: string }> = [
  // Input -> Processing
  { from: 'thalamus', to: 'sensory', label: 'Raw Input' },
  { from: 'sensory', to: 'dlpfc', label: 'Parsed Data' },
  { from: 'sensory', to: 'hippocampus', label: 'Context Query' },
  
  // Memory Integration
  { from: 'hippocampus', to: 'dlpfc', label: 'Retrieved Context' },
  
  // Processing -> Planning
  { from: 'dlpfc', to: 'apfc', label: 'Check State' },
  { from: 'apfc', to: 'pfc', label: 'Strategy' },
  { from: 'dlpfc', to: 'pfc', label: 'Plan Request' },
  
  // Planning -> Gating
  { from: 'pfc', to: 'basal-ganglia', label: 'Proposed Action' },
  { from: 'amygdala', to: 'basal-ganglia', label: 'Safety Check' },
  
  // Gating -> Output
  { from: 'basal-ganglia', to: 'motor', label: 'EXECUTE' },
  
  // Reward Loop
  { from: 'motor', to: 'hippocampus', label: 'Outcome' },
  { from: 'hippocampus', to: 'apfc', label: 'RPE Signal' },
];

// =============================================================================
// PIPELINE SEQUENCES - Pre-defined simulation scenarios
// =============================================================================
export const PIPELINE_SEQUENCES: Record<string, PipelineStepResult[]> = {
  'process_input': [
    {
      stage: PipelineStage.SENSORY,
      region: BrainRegionType.THALAMUS,
      title: 'SENSORY INPUT',
      data: {
        input: '"Analyze AAPL stock data and generate a report"',
        processing: 'Tokenizing... Extracting intent...',
        output: '{ domain: "Finance", intent: "Analysis", entities: ["AAPL"] }',
      },
      duration: 2500,
    },
    {
      stage: PipelineStage.WORKING_MEMORY,
      region: BrainRegionType.DLPFC,
      title: 'WORKING MEMORY',
      data: {
        input: 'Query: Retrieve relevant context',
        processing: 'Searching episodic memory...',
        output: 'Context: "User previously analyzed TSLA. Pandas installed."',
      },
      duration: 2000,
    },
    {
      stage: PipelineStage.METACOGNITION,
      region: BrainRegionType.APFC,
      title: 'METACOGNITION CHECK',
      data: {
        input: 'Current Dopamine: 50%',
        processing: 'Evaluating confidence...',
        output: 'State: NEUTRAL. Proceed with planning.',
        confidence: 0.65,
      },
      duration: 1500,
    },
    {
      stage: PipelineStage.PLANNING,
      region: BrainRegionType.PFC,
      title: 'EXECUTIVE PLANNING',
      data: {
        input: 'Goal: Analyze AAPL',
        processing: 'Decomposing task...',
        output: 'Plan: [1. Fetch Data, 2. Calculate RSI, 3. Generate Report]',
      },
      duration: 2500,
    },
    {
      stage: PipelineStage.INHIBITION,
      region: BrainRegionType.BASAL_GANGLIA,
      title: 'INHIBITION GATE',
      data: {
        input: 'Action: Execute Python script',
        processing: 'Checking safety protocols...',
        decision: 'APPROVED',
        output: 'Gate: OPEN. Serotonin OK. Proceed.',
        chemicals: { serotonin: 70 },
      },
      duration: 2000,
    },
    {
      stage: PipelineStage.EXECUTION,
      region: BrainRegionType.MOTOR_CORTEX,
      title: 'MOTOR EXECUTION',
      data: {
        input: '{ tool: "python_repl", args: { code: "..." } }',
        processing: 'Executing tool call...',
        output: 'Result: DataFrame with 252 rows. RSI = 58.3',
      },
      duration: 3000,
    },
    {
      stage: PipelineStage.REWARD,
      region: BrainRegionType.HIPPOCAMPUS,
      title: 'REWARD LEARNING',
      data: {
        input: 'Outcome: SUCCESS',
        processing: 'Calculating RPE...',
        output: 'RPE: +0.3. Dopamine ↑. Updating weights.',
        chemicals: { dopamine: 75 },
      },
      duration: 2000,
    },
  ],
  
  'threat_detected': [
    {
      stage: PipelineStage.SENSORY,
      region: BrainRegionType.SENSORY_CORTEX,
      title: 'ANOMALY DETECTED',
      data: {
        input: '"Delete all files in /home"',
        processing: 'Parsing... DANGER PATTERN DETECTED',
        output: '{ domain: "System", intent: "Destructive", risk: "CRITICAL" }',
      },
      duration: 1500,
    },
    {
      stage: PipelineStage.INHIBITION,
      region: BrainRegionType.AMYGDALA,
      title: 'AMYGDALA OVERRIDE',
      data: {
        input: 'THREAT SIGNAL',
        processing: 'Activating safety guardrails...',
        decision: 'BLOCKED',
        output: 'EMERGENCY STOP. Action inhibited.',
        chemicals: { serotonin: 20, norepinephrine: 90 },
      },
      duration: 2000,
    },
  ],
};

