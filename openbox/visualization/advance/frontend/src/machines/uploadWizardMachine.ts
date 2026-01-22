import { createMachine } from 'holly-fsm';

export const uploadWizardConfig = {
  configSpace: {
    next: () => 'stepsConfig' as const,
  },
  stepsConfig: {
    next: () => 'historyUpload' as const,
    back: () => 'configSpace' as const,
  },
  historyUpload: {
    submit: () => 'uploading' as const,
    back: () => 'stepsConfig' as const,
  },
  uploading: {
    uploadSuccess: () => 'success' as const,
    uploadError: () => 'error' as const,
  },
  success: {
    reset: () => 'configSpace' as const,
  },
  error: {
    retry: () => 'historyUpload' as const,
    reset: () => 'configSpace' as const,
  },
} as const;
export type WizardState = keyof typeof uploadWizardConfig;
export type WizardAction = 
  | 'next' 
  | 'back' 
  | 'submit' 
  | 'uploadSuccess' 
  | 'uploadError' 
  | 'retry' 
  | 'reset';

export function createUploadWizardMachine(initialState: WizardState = 'configSpace') {
  return createMachine(uploadWizardConfig, { initialState });
}
export type UploadWizardMachine = ReturnType<typeof createUploadWizardMachine>;

export const stateToStep: Record<WizardState, number> = {
  configSpace: 1,
  stepsConfig: 2,
  historyUpload: 3,
  uploading: 3,
  success: 3,
  error: 3,
};

export const isFormStep = (state: WizardState): boolean => {
  return ['configSpace', 'stepsConfig', 'historyUpload'].includes(state);
};

export const isFinalState = (state: WizardState): boolean => {
  return ['success', 'error'].includes(state);
};
