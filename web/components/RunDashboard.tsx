"use client";
import { useRun } from "../lib/store";
import { DashboardView } from "./Dashboard";

/** Client entry for /run/[id]: binds the shared run connection to the dashboard. */
export function RunDashboard({ runId }: { runId: string }) {
  const view = useRun(runId);
  return <DashboardView view={view} runId={runId} />;
}
