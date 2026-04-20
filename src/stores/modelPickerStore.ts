import { create } from "zustand";
import { ipc } from "../lib/ipc";
import type { ModelPickerPreference } from "../types";

const MODEL_PICKER_PREFERENCES_KEY = "panes.modelPicker.preferences.v1";

interface ModelPickerState {
  preferences: Record<string, ModelPickerPreference>;
  loaded: boolean;
  ensureLoaded: () => Promise<void>;
  toggleFavorite: (engineId: string, modelId: string) => Promise<void>;
  toggleEnabled: (engineId: string, modelId: string) => Promise<void>;
  getPreference: (engineId: string, modelId: string) => ModelPickerPreference;
}

function modelPreferenceKey(engineId: string, modelId: string): string {
  return `${engineId}::${modelId}`;
}

function normalizePreference(value: unknown): ModelPickerPreference | null {
  if (!value || typeof value !== "object") {
    return null;
  }

  const candidate = value as Partial<ModelPickerPreference>;
  return {
    favorite: candidate.favorite === true,
    enabled: candidate.enabled !== false,
  };
}

function normalizePreferences(
  value: unknown,
): Record<string, ModelPickerPreference> {
  if (!value || typeof value !== "object") {
    return {};
  }

  const next: Record<string, ModelPickerPreference> = {};
  for (const [key, pref] of Object.entries(value as Record<string, unknown>)) {
    const normalized = normalizePreference(pref);
    if (normalized) {
      next[key] = normalized;
    }
  }
  return next;
}

function readLocalPreferences(): Record<string, ModelPickerPreference> {
  try {
    const raw = localStorage.getItem(MODEL_PICKER_PREFERENCES_KEY);
    if (!raw) {
      return {};
    }
    return normalizePreferences(JSON.parse(raw));
  } catch {
    return {};
  }
}

function writeLocalPreferences(preferences: Record<string, ModelPickerPreference>): void {
  try {
    if (Object.keys(preferences).length === 0) {
      localStorage.removeItem(MODEL_PICKER_PREFERENCES_KEY);
      return;
    }
    localStorage.setItem(MODEL_PICKER_PREFERENCES_KEY, JSON.stringify(preferences));
  } catch {
    // Ignore persistence failures in tests or restricted environments.
  }
}

export const useModelPickerStore = create<ModelPickerState>((set, get) => ({
  preferences: {},
  loaded: false,
  ensureLoaded: async () => {
    if (get().loaded) {
      return;
    }

    const localPreferences = readLocalPreferences();
    set({
      preferences: localPreferences,
      loaded: true,
    });

    try {
      const remotePreferences = await ipc.getModelPickerPreferences();
      const normalizedRemote = normalizePreferences(remotePreferences);
      if (Object.keys(normalizedRemote).length === 0) {
        return;
      }
      set({
        preferences: normalizedRemote,
      });
      writeLocalPreferences(normalizedRemote);
    } catch {
      // IPC sync is best-effort for compatibility with older backends.
    }
  },
  toggleFavorite: async (engineId, modelId) => {
    const key = modelPreferenceKey(engineId, modelId);
    const current = get().preferences[key] ?? { favorite: false, enabled: true };
    const nextPreferences = {
      ...get().preferences,
      [key]: {
        ...current,
        favorite: !current.favorite,
      },
    };
    set({ preferences: nextPreferences });
    writeLocalPreferences(nextPreferences);
    try {
      await ipc.setModelPickerPreferences(nextPreferences);
    } catch {
      // IPC sync is best-effort for compatibility with older backends.
    }
  },
  toggleEnabled: async (engineId, modelId) => {
    const key = modelPreferenceKey(engineId, modelId);
    const current = get().preferences[key] ?? { favorite: false, enabled: true };
    const nextPreferences = {
      ...get().preferences,
      [key]: {
        ...current,
        enabled: !current.enabled,
      },
    };
    set({ preferences: nextPreferences });
    writeLocalPreferences(nextPreferences);
    try {
      await ipc.setModelPickerPreferences(nextPreferences);
    } catch {
      // IPC sync is best-effort for compatibility with older backends.
    }
  },
  getPreference: (engineId, modelId) => {
    const key = modelPreferenceKey(engineId, modelId);
    return get().preferences[key] ?? { favorite: false, enabled: true };
  },
}));
