// =============================================================================
// NEURO-MIMETIC CORE - TYPE DEFINITIONS
// =============================================================================

// --- Brain Region Types (Anatomically Accurate for PFC Pipeline) ---
export enum BrainRegionType {
  // Sensory Input
  THALAMUS = 'Thalamus',
  SENSORY_CORTEX = 'Sensory Cortex',
  
  // Processing (PFC Sub-regions)
  PFC = 'Prefrontal Cortex',
  DLPFC = 'dlPFC',
  APFC = 'aPFC',
  
  // Decision/Gating
  BASAL_GANGLIA = 'Basal Ganglia',
  AMYGDALA = 'Amygdala',
  
  // Output
  MOTOR_CORTEX = 'Motor Cortex',
  
  // Memory
  HIPPOCAMPUS = 'Hippocampus',
}

// --- Pipeline Stage (matches our PFC architecture) ---
export enum PipelineStage {
  IDLE = 'IDLE',
  SENSORY = 'SENSORY_INPUT',
  WORKING_MEMORY = 'WORKING_MEMORY',
  METACOGNITION = 'METACOGNITION',
  PLANNING = 'PLANNING',
  INHIBITION = 'INHIBITION_GATE',
  EXECUTION = 'MOTOR_EXECUTION',
  REWARD = 'REWARD_LEARNING',
}

// --- Neurochemistry State ---
export interface NeuroState {
  dopamine: number;      // 0-100: Reward/Motivation
  serotonin: number;     // 0-100: Stability/Caution
  norepinephrine: number; // 0-100: Alertness/Focus
}

// --- Brain Node Data for 3D Visualization ---
export interface BrainNodeData {
  id: string;
  type: BrainRegionType;
  position: [number, number, number];
  color: string;
  label: string;
  description: string;
  pipelineStage?: PipelineStage; // Which pipeline stage this region handles
}

// --- Pipeline Step Result (JSON displayed at each stage) ---
export interface PipelineStepResult {
  stage: PipelineStage;
  region: BrainRegionType;
  title: string;
  data: {
    input?: string;
    processing?: string;
    output?: string;
    decision?: 'APPROVED' | 'BLOCKED' | 'PENDING';
    confidence?: number;
    chemicals?: Partial<NeuroState>;
  };
  duration: number; // ms to display this step
}

// --- Log Entry ---
export interface LogEntry {
  id: string;
  timestamp: number;
  message: string;
  type: 'info' | 'success' | 'warning' | 'error';
  stage?: PipelineStage;
  region?: BrainRegionType;
}

// --- Active Processing State ---
export interface ProcessingState {
  isRunning: boolean;
  currentStage: PipelineStage;
  currentRegion: BrainRegionType | null;
  currentData: PipelineStepResult | null;
  progress: number; // 0-100
}
