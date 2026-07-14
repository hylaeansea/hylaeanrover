import { createContext, useContext } from "react";
import type { Schema } from "./types";

export interface StudioActions {
  updateParam: (
    nodeId: string,
    name: string,
    value: number | string | boolean | null,
  ) => void;
  runNode: (nodeId: string) => void;
  stopNode: (nodeId: string) => void;
  launchSim: (onnx: string) => void;
  openRun: (runId: string) => void;
}

export interface StudioContextValue {
  schema: Schema;
  actions: StudioActions;
}

export const StudioContext = createContext<StudioContextValue | null>(null);

export function useStudio(): StudioContextValue {
  const ctx = useContext(StudioContext);
  if (!ctx) throw new Error("StudioContext missing");
  return ctx;
}
