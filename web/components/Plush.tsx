/**
 * The plush crew mascots: one soft-bodied, cold-tech "mecha plush" per agent role, drawn as inline
 * SVG (no image assets, no network). Frosted steel shells, LED visor eyes, panel seams and one neon
 * accent per role; each role keeps a distinct creature and a job prop.
 *
 * Purely decorative (aria-hidden): the crew card next to each one carries the text equivalent.
 * Motion is CSS only, keyed off `data-pose` (styles/terminal.css, "plush crew"), and only on
 * transform / opacity of SVG groups. Under prefers-reduced-motion every animation is off and the
 * pose is a static stance.
 */
import type { ReactNode } from "react";

export const COORDINATOR = "coordinator";

/** The sub-agent roles of src/posttrain_agent/subagents.py, in pipeline order. */
export const ROLES = [
  "benchmark",
  "failure_analyst",
  "researcher",
  "data_curator",
  "training_designer",
  "training_executor",
  "evaluator",
  "champion_manager",
] as const;

/**
 * asleep:  never seen in agent events nor in any A2A message.
 * standby: no agent events, but it appears in the A2A conversation (awake, not running).
 * idle:    its agent ended without an ok flag (or the coordinator outside RUNNING).
 * working: agent.start seen and no agent.end yet.
 * done / failed: agent.end with ok true / false.
 */
export type Pose = "asleep" | "standby" | "idle" | "working" | "done" | "failed";

type Look = { creature: string; accent: string; prop: string };

/** Creature, LED accent and prop per role. Unknown roles get the generic drone. */
export const LOOKS: Record<string, Look> = {
  coordinator: { creature: "star core", accent: "#8ff6ff", prop: "crown and signal wand" },
  benchmark: { creature: "frog", accent: "#b6ff3b", prop: "stopwatch" },
  failure_analyst: { creature: "hedgehog", accent: "#ff5cf0", prop: "clipboard" },
  researcher: { creature: "owl", accent: "#b18cff", prop: "magnifier" },
  data_curator: { creature: "fox", accent: "#ff9a3d", prop: "data stack" },
  training_designer: { creature: "sheep", accent: "#5aa9ff", prop: "blueprint" },
  training_executor: { creature: "robot bunny", accent: "#3ef0ff", prop: "wrench" },
  evaluator: { creature: "cat", accent: "#7dffcf", prop: "scorecard" },
  champion_manager: { creature: "bear", accent: "#ffd84a", prop: "trophy" },
};
const DRONE: Look = { creature: "drone", accent: "#9fb4d8", prop: "none" };
export const lookOf = (role: string): Look => (Object.prototype.hasOwnProperty.call(LOOKS, role) ? LOOKS[role] : DRONE);

const SHELL = "url(#pc-shell) #8a9bb5";
const DARK = "url(#pc-dark) #2a3550";
const VISOR = "url(#pc-visor) #0b1022";
const FROST = "url(#pc-frost) #c9d6ea";
const LINE = "#0a1024";
const BLOB = "M50 24 C73 24 84 41 84 61 C84 80 70 90 50 90 C30 90 16 80 16 61 C16 41 27 24 50 24 Z";

/**
 * Gradients shared by every plush (rendered once by the crew). Each paint above also names a flat
 * fallback colour, so a plush rendered without these defs still draws.
 */
export function PlushDefs() {
  return (
    <svg width="0" height="0" aria-hidden="true" focusable="false" style={{ position: "absolute" }}>
      <defs>
        <linearGradient id="pc-shell" x1="0.15" y1="0" x2="0.85" y2="1">
          <stop offset="0" stopColor="#eef4ff" />
          <stop offset="0.45" stopColor="#a9b8d2" />
          <stop offset="1" stopColor="#3d4a66" />
        </linearGradient>
        <linearGradient id="pc-dark" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0" stopColor="#56658a" />
          <stop offset="1" stopColor="#161d33" />
        </linearGradient>
        <linearGradient id="pc-visor" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stopColor="#1a2550" />
          <stop offset="1" stopColor="#050916" />
        </linearGradient>
        <radialGradient id="pc-frost" cx="0.35" cy="0.3" r="0.8">
          <stop offset="0" stopColor="#ffffff" />
          <stop offset="0.6" stopColor="#c6d3ea" />
          <stop offset="1" stopColor="#7d8db0" />
        </radialGradient>
      </defs>
    </svg>
  );
}

function Shell({ d = BLOB, fill = SHELL }: { d?: string; fill?: string }) {
  return (
    <>
      <path d={d} fill={fill} stroke={LINE} strokeWidth={1.4} />
      {/* frosted highlight + panel seam + accent "circuit stitch" */}
      <ellipse cx={37} cy={36} rx={11} ry={6} fill="#ffffff" opacity={0.28} transform="rotate(-24 37 36)" />
      <path d="M22 67 Q50 77 78 67" fill="none" stroke="#24324f" strokeWidth={1} opacity={0.75} />
      <path d="M26 72 Q50 82 74 72" fill="none" stroke="currentColor" strokeWidth={1.1} strokeDasharray="1.5 3" opacity={0.75} />
    </>
  );
}

function Visor({ y = 46 }: { y?: number }) {
  return (
    <>
      <rect x={26} y={y} width={48} height={14} rx={7} fill={VISOR} stroke="currentColor" strokeOpacity={0.45} />
      <g className="p-eyes">
        <rect className="p-led" x={35} y={y + 4.5} width={9} height={5} rx={2.5} fill="currentColor" />
        <rect className="p-led" x={56} y={y + 4.5} width={9} height={5} rx={2.5} fill="currentColor" />
      </g>
    </>
  );
}

function Limbs() {
  return (
    <>
      <ellipse cx={38} cy={89} rx={8} ry={3.5} fill={DARK} />
      <ellipse cx={62} cy={89} rx={8} ry={3.5} fill={DARK} />
      <ellipse cx={18} cy={67} rx={4.5} ry={8} fill={DARK} transform="rotate(18 18 67)" />
      <ellipse cx={82} cy={67} rx={4.5} ry={8} fill={DARK} transform="rotate(-18 82 67)" />
    </>
  );
}

function Core({ cy = 79 }: { cy?: number }) {
  return <circle className="p-led p-core" cx={50} cy={cy} r={3} fill="currentColor" />;
}

/** Per role: what sits behind the shell, what sits on it, and the job prop. */
type Parts = { back?: ReactNode; face?: ReactNode; prop?: ReactNode; shellFill?: string; shellPath?: string };

const STAR =
  "50,20 64.1,38.6 86.1,46.3 72.8,65.4 72.3,88.7 50,82 27.7,88.7 27.2,65.4 13.9,46.3 35.9,38.6";

function parts(role: string, trophyLit: boolean): Parts {
  switch (role) {
    case COORDINATOR:
      return {
        face: (
          <>
            <polygon points="39,17 41,7 45.5,13 50,4 54.5,13 59,7 61,17" fill={SHELL} stroke={LINE} strokeWidth={1.2} strokeLinejoin="round" />
            <circle className="p-led" cx={50} cy={9} r={1.8} fill="currentColor" />
            <Visor y={47} />
            <Core cy={70} />
          </>
        ),
        prop: (
          <g className="p-arm">
            <rect x={84} y={44} width={5} height={18} rx={2.5} fill={DARK} transform="rotate(-35 86 53)" />
            <line x1={90} y1={47} x2={96} y2={38} stroke="#c9d6ea" strokeWidth={1.6} strokeLinecap="round" />
            <circle className="p-led" cx={96.5} cy={37} r={2.4} fill="currentColor" />
          </g>
        ),
      };
    case "benchmark":
      return {
        back: (
          <>
            <circle cx={34} cy={30} r={10} fill={SHELL} stroke={LINE} strokeWidth={1.4} />
            <circle cx={66} cy={30} r={10} fill={SHELL} stroke={LINE} strokeWidth={1.4} />
          </>
        ),
        face: (
          <>
            <circle cx={34} cy={30} r={6} fill={VISOR} />
            <circle cx={66} cy={30} r={6} fill={VISOR} />
            <g className="p-eyes">
              <circle className="p-led" cx={34} cy={30} r={2.8} fill="currentColor" />
              <circle className="p-led" cx={66} cy={30} r={2.8} fill="currentColor" />
            </g>
            <path d="M36 56 Q50 61 64 56" fill="none" stroke="currentColor" strokeWidth={1.4} strokeLinecap="round" opacity={0.8} />
            <Core />
          </>
        ),
        prop: (
          <g className="p-prop">
            <rect x={79} y={52} width={4} height={3} fill="#c9d6ea" />
            <circle cx={81} cy={64} r={8} fill={VISOR} stroke="#c9d6ea" strokeWidth={2} />
            <line className="p-hand" x1={81} y1={64} x2={81} y2={58.5} stroke="currentColor" strokeWidth={1.6} strokeLinecap="round" />
          </g>
        ),
      };
    case "failure_analyst":
      return {
        back: (
          <polygon
            points="18,60 12,47 23,44 20,31 32,33 35,20 45,27 50,14 55,27 65,20 68,33 80,31 77,44 88,47 82,60"
            fill={DARK}
            stroke={LINE}
            strokeWidth={1.2}
            strokeLinejoin="round"
          />
        ),
        face: (
          <>
            <Visor />
            <ellipse cx={50} cy={65} rx={4} ry={3} fill={DARK} />
            <Core />
          </>
        ),
        prop: (
          <g className="p-prop">
            <rect x={5} y={55} width={15} height={20} rx={2} fill="#dbe5f5" stroke={LINE} strokeWidth={1} />
            <rect x={9} y={53} width={7} height={4} rx={1} fill={DARK} />
            <path d="M8 62 h9 M8 66 h9 M8 70 h6" stroke="currentColor" strokeWidth={1.3} />
          </g>
        ),
      };
    case "researcher":
      return {
        back: (
          <>
            <polygon points="27,33 22,13 41,27" fill={SHELL} stroke={LINE} strokeWidth={1.2} strokeLinejoin="round" />
            <polygon points="73,33 78,13 59,27" fill={SHELL} stroke={LINE} strokeWidth={1.2} strokeLinejoin="round" />
          </>
        ),
        face: (
          <>
            <circle cx={38} cy={50} r={9.5} fill={VISOR} stroke="currentColor" strokeOpacity={0.55} />
            <circle cx={62} cy={50} r={9.5} fill={VISOR} stroke="currentColor" strokeOpacity={0.55} />
            <g className="p-eyes">
              <circle className="p-led" cx={38} cy={50} r={3.4} fill="currentColor" />
              <circle className="p-led" cx={62} cy={50} r={3.4} fill="currentColor" />
            </g>
            <polygon points="47,59 53,59 50,64" fill={DARK} />
            <Core />
          </>
        ),
        prop: (
          <g className="p-prop">
            <line x1={86} y1={70} x2={92} y2={78} stroke={DARK} strokeWidth={3.4} strokeLinecap="round" />
            <circle cx={81} cy={63} r={7.5} fill="currentColor" fillOpacity={0.18} stroke="#c9d6ea" strokeWidth={2.4} />
          </g>
        ),
      };
    case "data_curator":
      return {
        back: (
          <>
            <polygon points="24,36 21,11 43,27" fill={SHELL} stroke={LINE} strokeWidth={1.2} strokeLinejoin="round" />
            <polygon points="76,36 79,11 57,27" fill={SHELL} stroke={LINE} strokeWidth={1.2} strokeLinejoin="round" />
            <polygon points="26,30 24,17 36,26" fill="currentColor" opacity={0.55} />
            <polygon points="74,30 76,17 64,26" fill="currentColor" opacity={0.55} />
          </>
        ),
        face: (
          <>
            <path d="M33 60 Q50 82 67 60 Q50 67 33 60 Z" fill="#eef4ff" opacity={0.85} />
            <Visor />
            <circle cx={50} cy={66} r={2.2} fill={LINE} />
            <Core cy={80} />
          </>
        ),
        prop: (
          <g className="p-prop">
            <rect x={3} y={73} width={19} height={5} rx={1.5} fill={VISOR} stroke="currentColor" strokeWidth={0.9} />
            <rect x={4} y={67} width={18} height={5} rx={1.5} fill={VISOR} stroke="currentColor" strokeWidth={0.9} />
            <rect x={5} y={61} width={16} height={5} rx={1.5} fill={VISOR} stroke="currentColor" strokeWidth={0.9} />
          </g>
        ),
      };
    case "training_designer":
      return {
        shellFill: DARK,
        back: (
          <>
            {[
              [24, 34],
              [36, 24],
              [50, 20],
              [64, 24],
              [76, 34],
              [84, 50],
              [16, 50],
            ].map(([cx, cy]) => (
              <circle key={`${cx}-${cy}`} cx={cx} cy={cy} r={11} fill={FROST} stroke={LINE} strokeWidth={1} />
            ))}
          </>
        ),
        face: (
          <>
            <Visor />
            <Core />
          </>
        ),
        prop: (
          <g className="p-prop">
            <rect x={72} y={55} width={22} height={16} rx={1.5} fill="#0f2c63" stroke="currentColor" strokeWidth={1} transform="rotate(8 83 63)" />
            <path d="M76 60 h14 M76 64 h14 M80 57 v12 M86 57 v12" stroke="currentColor" strokeWidth={0.6} opacity={0.8} transform="rotate(8 83 63)" />
          </g>
        ),
      };
    case "training_executor":
      return {
        back: (
          <>
            <rect x={29} y={0} width={11} height={34} rx={5.5} fill={SHELL} stroke={LINE} strokeWidth={1.2} />
            <rect x={60} y={0} width={11} height={34} rx={5.5} fill={SHELL} stroke={LINE} strokeWidth={1.2} />
            <rect x={32.5} y={5} width={4} height={20} rx={2} fill="currentColor" opacity={0.5} />
            <rect x={63.5} y={5} width={4} height={20} rx={2} fill="currentColor" opacity={0.5} />
          </>
        ),
        face: (
          <>
            <Visor />
            <rect x={44} y={64} width={12} height={3} rx={1.5} fill={DARK} />
            <Core />
          </>
        ),
        prop: (
          <g className="p-prop">
            <path
              d="M80 76 L88 58 M84.5 53 a5 5 0 1 0 7 5 l-3 -1 -1.5 -3.5 z"
              fill="#c9d6ea"
              stroke="#c9d6ea"
              strokeWidth={3}
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </g>
        ),
      };
    case "evaluator":
      return {
        back: (
          <>
            <polygon points="21,38 25,15 41,29" fill={SHELL} stroke={LINE} strokeWidth={1.2} strokeLinejoin="round" />
            <polygon points="79,38 75,15 59,29" fill={SHELL} stroke={LINE} strokeWidth={1.2} strokeLinejoin="round" />
          </>
        ),
        face: (
          <>
            <Visor />
            <path d="M24 60 h-12 M24 64 l-11 3 M76 60 h12 M76 64 l11 3" stroke="currentColor" strokeWidth={0.9} opacity={0.8} />
            <polygon points="47,62 53,62 50,65" fill={DARK} />
            <Core />
          </>
        ),
        prop: (
          <g className="p-prop">
            <rect x={4} y={55} width={16} height={21} rx={2} fill="#e6eefb" stroke={LINE} strokeWidth={1} />
            <path d="M7 61 h5 M7 66 h5 M7 71 h5" stroke={DARK} strokeWidth={1.4} />
            <path d="M14 60 l1.5 1.8 3 -3.6 M14 65 l1.5 1.8 3 -3.6" fill="none" stroke="currentColor" strokeWidth={1.3} />
          </g>
        ),
      };
    case "champion_manager":
      return {
        back: (
          <>
            <circle cx={28} cy={30} r={9} fill={SHELL} stroke={LINE} strokeWidth={1.2} />
            <circle cx={72} cy={30} r={9} fill={SHELL} stroke={LINE} strokeWidth={1.2} />
            <circle cx={28} cy={30} r={4.2} fill={DARK} />
            <circle cx={72} cy={30} r={4.2} fill={DARK} />
          </>
        ),
        face: (
          <>
            <Visor />
            <ellipse cx={50} cy={66} rx={8} ry={5} fill="#dfe8f5" opacity={0.85} />
            <ellipse cx={50} cy={64.5} rx={2.6} ry={1.8} fill={LINE} />
            <Core cy={81} />
          </>
        ),
        prop: (
          // The trophy is steel; it lights up only after a PROMOTE gate verdict.
          <g className="p-prop p-trophy" data-lit={trophyLit ? "true" : "false"}>
            <path
              d="M76 54 h14 v4 a7 7 0 0 1 -14 0 z M81 65 h4 v5 h3 v3 h-10 v-3 h3 z"
              fill={trophyLit ? "currentColor" : "#9aa8c2"}
              stroke={LINE}
              strokeWidth={0.9}
            />
          </g>
        ),
      };
    default:
      return {
        face: (
          <>
            <Visor />
            <Core />
          </>
        ),
      };
  }
}

type PlushProps = {
  role: string;
  pose: Pose;
  /** champion_manager only: light the trophy (a PROMOTE verdict is on screen). */
  trophyLit?: boolean;
  /** Bumped (as a React key) when the coordinator just sent a brief: replays the point gesture. */
  gesture?: number | null;
  className?: string;
};

/** One mascot. Decorative: aria-hidden, the caller renders the text equivalent. */
export function Plush({ role, pose, trophyLit = false, gesture = null, className }: PlushProps) {
  const look = lookOf(role);
  const p = parts(role, trophyLit);
  const coord = role === COORDINATOR;
  return (
    <svg
      className={`plush plush-${coord ? "coordinator" : "agent"}${className ? ` ${className}` : ""}`}
      viewBox="0 0 100 100"
      data-pose={pose}
      data-role={role}
      data-gesture={gesture !== null ? "true" : "false"}
      aria-hidden="true"
      focusable="false"
      style={{ color: look.accent }}
    >
      <ellipse className="p-shadow" cx={50} cy={93} rx={28} ry={4} fill="#000" opacity={0.45} />
      <g className="p-hop">
        <g className="p-breathe">
          <g className="p-body">
            {p.back}
            <Limbs />
            {coord ? (
              <>
                <polygon points={STAR} fill="none" stroke={LINE} strokeWidth={12} strokeLinejoin="round" />
                <polygon points={STAR} fill={SHELL} stroke={SHELL} strokeWidth={9} strokeLinejoin="round" />
                <ellipse cx={40} cy={40} rx={9} ry={5} fill="#ffffff" opacity={0.3} transform="rotate(-24 40 40)" />
                <path d="M30 76 Q50 86 70 76" fill="none" stroke="currentColor" strokeWidth={1.1} strokeDasharray="1.5 3" opacity={0.75} />
              </>
            ) : (
              <Shell d={p.shellPath} fill={p.shellFill} />
            )}
            {p.face}
          </g>
          {/* keyed: a new gesture value remounts the arm, replaying its finite animation */}
          <g key={gesture ?? "rest"} className="p-propwrap">
            {p.prop}
          </g>
        </g>
      </g>
      <g className="p-fx">
        <path className="p-spark" d="M84 18 l2 5 5 2 -5 2 -2 5 -2 -5 -5 -2 5 -2 z" fill="currentColor" />
        <path className="p-spark p-spark-2" d="M15 30 l1.4 3.4 3.4 1.4 -3.4 1.4 -1.4 3.4 -1.4 -3.4 -3.4 -1.4 3.4 -1.4 z" fill="currentColor" />
        <g className="p-zzz" fill="none" stroke="#9fb4d8" strokeWidth={1.6} strokeLinecap="round" strokeLinejoin="round">
          <path d="M70 26 h6 l-6 7 h6" />
          <path d="M80 14 h4 l-4 5 h4" />
        </g>
        <path className="p-drop" d="M78 34 q4 6 0 9 q-4 -3 0 -9 z" fill="#8fd8ff" stroke={LINE} strokeWidth={0.6} />
      </g>
    </svg>
  );
}
