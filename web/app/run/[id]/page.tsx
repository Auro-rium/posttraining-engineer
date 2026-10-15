import { RunDashboard } from "../../../components/RunDashboard";

function decode(id: string): string {
  try {
    return decodeURIComponent(id);
  } catch {
    return id;
  }
}

export default async function RunPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  return <RunDashboard runId={decode(id)} />;
}
