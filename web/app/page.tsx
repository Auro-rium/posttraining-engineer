import { Boot } from "../components/Boot";
import { RunList } from "../components/RunList";

export default function Home() {
  return (
    <div className="home-shell">
      <div className="topline">
        <Boot text="> mission control :: run index" />
      </div>
      <main className="home">
        <h1>[ POSTTRAIN :: runs ]</h1>
        <RunList />
      </main>
    </div>
  );
}
