import { RunList } from "../components/RunList";

export default function Home() {
  return (
    <main className="home">
      <h1>[ POSTTRAIN :: runs ]</h1>
      <RunList />
    </main>
  );
}
