"use client";
import { useCallback, useEffect, useRef, useState } from "react";

export type BlipKind = "tick" | "alert" | "good";

/** Minimum gap between two blips: a burst of events makes one sound, not a buzz. */
export const BLIP_GAP_MS = 140;

type AudioCtor = new () => AudioContext;

function audioCtor(): AudioCtor | null {
  if (typeof window === "undefined") return null;
  const w = window as unknown as { AudioContext?: AudioCtor; webkitAudioContext?: AudioCtor };
  return w.AudioContext ?? w.webkitAudioContext ?? null;
}

const TONE: Record<BlipKind, { hz: number; ms: number; type: OscillatorType }> = {
  tick: { hz: 1320, ms: 28, type: "square" },
  good: { hz: 880, ms: 90, type: "triangle" },
  alert: { hz: 220, ms: 140, type: "sawtooth" },
};

/**
 * Optional terminal blips. Off by default and never persisted: the AudioContext is created only
 * inside the click that turns sound on (a user gesture), never on load or on an event.
 */
export function useBlips() {
  const [on, setOn] = useState(false);
  // Becomes false only after a click found no usable Web Audio (keeps server and client markup equal).
  const [supported, setSupported] = useState(true);
  const onRef = useRef(false);
  const ctx = useRef<AudioContext | null>(null);
  const lastAt = useRef(0);

  const toggle = useCallback(() => {
    if (onRef.current) {
      onRef.current = false;
      setOn(false);
      void ctx.current?.suspend?.()?.catch?.(() => {});
      return;
    }
    if (!ctx.current) {
      const Ctor = audioCtor();
      try {
        ctx.current = Ctor ? new Ctor() : null;
      } catch {
        ctx.current = null;
      }
      if (!ctx.current) {
        setSupported(false);
        return;
      }
    }
    void ctx.current.resume?.()?.catch?.(() => {});
    onRef.current = true;
    setOn(true);
  }, []);

  const blip = useCallback(
    (kind: BlipKind) => {
      const ac = ctx.current;
      if (!on || !ac) return;
      const now = Date.now();
      if (now - lastAt.current < BLIP_GAP_MS) return;
      lastAt.current = now;
      try {
        const t = TONE[kind];
        const osc = ac.createOscillator();
        const gain = ac.createGain();
        osc.type = t.type;
        osc.frequency.value = t.hz;
        const start = ac.currentTime;
        gain.gain.setValueAtTime(0.0001, start);
        gain.gain.exponentialRampToValueAtTime(0.05, start + 0.005);
        gain.gain.exponentialRampToValueAtTime(0.0001, start + t.ms / 1000);
        osc.connect(gain).connect(ac.destination);
        osc.start(start);
        osc.stop(start + t.ms / 1000 + 0.02);
      } catch {
        /* audio is decoration: never let it break the dashboard */
      }
    },
    [on],
  );

  useEffect(
    () => () => {
      void ctx.current?.close?.()?.catch?.(() => {});
      ctx.current = null;
    },
    [],
  );

  return { on, toggle, blip, supported };
}

export function SoundToggle({ on, onToggle, supported = true }: { on: boolean; onToggle(): void; supported?: boolean }) {
  return (
    <button
      type="button"
      className="sound-toggle"
      aria-label="Terminal sound blips"
      aria-pressed={on}
      disabled={!supported}
      title={supported ? (on ? "Turn blips off" : "Turn blips on") : "Sound is not available in this browser"}
      onClick={onToggle}
    >
      <span aria-hidden="true">{on ? "♪" : "∅"}</span> blips {on ? "on" : "off"}
    </button>
  );
}
