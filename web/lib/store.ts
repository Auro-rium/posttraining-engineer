/**
 * React binding for the run connection: `useRun(runId)` via useSyncExternalStore.
 *
 * One shared connection per run id, ref-counted: the first subscriber starts it, the last one
 * stops it (closes the EventSource, clears timers, aborts fetches). A remount (or React strict
 * mode's subscribe/unsubscribe/subscribe) resumes from the last folded seq. Nothing here touches
 * window or EventSource at import time or during server render.
 */
import { useSyncExternalStore } from "react";
import { createConnection, type Connection, type RunView } from "./connection";

export type { Conn, Mode, RunView, Source } from "./connection";

export type RunStore = {
  subscribe(listener: () => void): () => void;
  getSnapshot(): RunView;
  getServerSnapshot(): RunView;
};

/** Wrap a connection so subscriptions drive its lifecycle (start on first, stop on last). */
export function createRunStore(conn: Connection): RunStore {
  let refs = 0;
  return {
    subscribe(listener) {
      const unsubscribe = conn.subscribe(listener);
      if (++refs === 1) conn.start();
      let active = true;
      return () => {
        if (!active) return;
        active = false;
        unsubscribe();
        if (--refs === 0) conn.stop();
      };
    },
    getSnapshot: () => conn.getSnapshot(),
    getServerSnapshot: () => conn.getServerSnapshot(),
  };
}

const registry = new Map<string, RunStore>();

/** The shared store for a run, configured from NEXT_PUBLIC_API_URL / NEXT_PUBLIC_READ_TOKEN. */
export function getRunStore(runId: string): RunStore {
  let store = registry.get(runId);
  if (!store) {
    store = createRunStore(
      createConnection({
        runId,
        // Written out literally so Next.js inlines them at build time.
        apiUrl: process.env.NEXT_PUBLIC_API_URL ?? null,
        token: process.env.NEXT_PUBLIC_READ_TOKEN ?? null,
      }),
    );
    registry.set(runId, store);
  }
  return store;
}

export function useRun(runId: string): RunView {
  const store = getRunStore(runId);
  return useSyncExternalStore(store.subscribe, store.getSnapshot, store.getServerSnapshot);
}
