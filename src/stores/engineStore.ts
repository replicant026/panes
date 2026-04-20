import { create } from "zustand";
import type {
  EngineHealth,
  EngineInfo,
  EngineModel,
  EngineRuntimeUpdatedEvent,
  ModelPreference,
} from "../types";
import { ipc } from "../lib/ipc";

interface EngineState {
  engines: EngineInfo[];
  activeWorkspaceId: string | null;
  modelPreferences: Record<string, ModelPreference>;
  health: Record<string, EngineHealth>;
  healthLoading: Record<string, boolean>;
  loading: boolean;
  loadedOnce: boolean;
  error?: string;
  load: () => Promise<void>;
  loadModelPreferences: (workspaceId: string, userId?: string | null) => Promise<void>;
  saveModelPreference: (
    workspaceId: string,
    engineId: string,
    modelId: string,
    patch: { isFavorite: boolean; isEnabled: boolean },
    userId?: string | null,
  ) => Promise<void>;
  ensureHealth: (
    engineId: string,
    options?: { force?: boolean },
  ) => Promise<EngineHealth | null>;
  mergeHealth: (reports: EngineHealth[]) => void;
  applyRuntimeUpdate: (event: EngineRuntimeUpdatedEvent) => void;
}

let pendingHealthRequests: Partial<Record<string, Promise<EngineHealth | null>>> = {};

export const useEngineStore = create<EngineState>((set, get) => ({
  engines: [],
  activeWorkspaceId: null,
  modelPreferences: {},
  health: {},
  healthLoading: {},
  loading: false,
  loadedOnce: false,
  load: async () => {
    set({ loading: true, error: undefined });
    try {
      const engines = applyModelPreferences(
        await ipc.listEngines(),
        get().modelPreferences,
      );
      set({
        engines,
        loading: false,
        loadedOnce: true,
        error: undefined,
      });
    } catch (error) {
      const message = String(error);
      set({
        loading: false,
        loadedOnce: true,
        error: message,
        health: {
          codex: {
            id: "codex",
            available: false,
            details: `Engine discovery failed: ${message}`,
            warnings: [],
            checks: ["codex --version", "command -v codex"],
            fixes: [],
          },
        },
      });
    }
  },
  loadModelPreferences: async (workspaceId, userId) => {
    const normalizedWorkspaceId = workspaceId.trim();
    if (!normalizedWorkspaceId) {
      set({ activeWorkspaceId: null, modelPreferences: {} });
      return;
    }
    const preferences = await ipc.getModelPreferences(normalizedWorkspaceId, userId ?? null);
    const preferenceMap = Object.fromEntries(
      preferences.map((value) => [modelPreferenceKey(value.engineId, value.modelId), value]),
    );
    set((state) => ({
      activeWorkspaceId: normalizedWorkspaceId,
      modelPreferences: preferenceMap,
      engines: applyModelPreferences(state.engines, preferenceMap),
    }));
  },
  saveModelPreference: async (workspaceId, engineId, modelId, patch, userId) => {
    const preference = await ipc.saveModelPreference(
      workspaceId,
      engineId,
      modelId,
      patch.isFavorite,
      patch.isEnabled,
      userId ?? null,
    );
    set((state) => {
      if (state.activeWorkspaceId !== workspaceId) {
        return state;
      }
      const nextPreferences = {
        ...state.modelPreferences,
        [modelPreferenceKey(engineId, modelId)]: preference,
      };
      return {
        modelPreferences: nextPreferences,
        engines: applyModelPreferences(state.engines, nextPreferences),
      };
    });
  },
  ensureHealth: async (engineId, options) => {
    const existing = get().health[engineId];
    if (existing && !options?.force) {
      return existing;
    }

    if (pendingHealthRequests[engineId]) {
      return pendingHealthRequests[engineId];
    }

    set((state) => {
      if (
        state.healthLoading[engineId] ||
        (!options?.force && state.health[engineId])
      ) {
        return state;
      }

      return {
        healthLoading: {
          ...state.healthLoading,
          [engineId]: true,
        },
      };
    });

    const request = (async () => {
      try {
        const health = await ipc.engineHealth(engineId);
        set((state) => {
          const { [engineId]: _ignored, ...rest } = state.healthLoading;
          return {
            health: {
              ...state.health,
              [health.id]: health,
            },
            healthLoading: rest,
          };
        });
        return health;
      } catch (error) {
        const message = String(error);
        set((state) => {
          const { [engineId]: _ignored, ...rest } = state.healthLoading;
          return {
            healthLoading: rest,
            error: `${engineId}: ${message}`,
          };
        });
        return null;
      } finally {
        delete pendingHealthRequests[engineId];
      }
    })();

    pendingHealthRequests[engineId] = request;
    return request;
  },
  mergeHealth: (reports) =>
    set((state) => {
      if (reports.length === 0) {
        return state;
      }

      const nextHealth = { ...state.health };
      const nextHealthLoading = { ...state.healthLoading };
      for (const report of reports) {
        nextHealth[report.id] = report;
        delete nextHealthLoading[report.id];
      }

      return {
        health: nextHealth,
        healthLoading: nextHealthLoading,
      };
    }),
  applyRuntimeUpdate: ({ engineId, protocolDiagnostics }) =>
    set((state) => {
      const current = state.health[engineId];
      const nextHealth: EngineHealth = current
        ? {
            ...current,
            available: true,
            details: current.available ? current.details : undefined,
            protocolDiagnostics: protocolDiagnostics ?? current.protocolDiagnostics,
          }
        : {
            id: engineId,
            available: true,
            warnings: [],
            checks: [],
            fixes: [],
            protocolDiagnostics,
          };

      const { [engineId]: _ignored, ...rest } = state.healthLoading;

      return {
        health: {
          ...state.health,
          [engineId]: nextHealth,
        },
        healthLoading: rest,
      };
    }),
}));

function modelPreferenceKey(engineId: string, modelId: string): string {
  return `${engineId}::${modelId}`;
}

function applyModelPreferences(
  engines: EngineInfo[],
  preferences: Record<string, ModelPreference>,
): EngineInfo[] {
  return engines.map((engine) => ({
    ...engine,
    models: engine.models.map((model): EngineModel => {
      const preference = preferences[modelPreferenceKey(engine.id, model.id)];
      return {
        ...model,
        isFavorite: preference?.isFavorite ?? false,
        isEnabled: preference?.isEnabled ?? true,
      };
    }),
  }));
}
