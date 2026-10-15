import type { ReactNode } from "react";

export const metadata = {
  title: "posttrain live",
  description: "Live dashboard for the post-training agent",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
