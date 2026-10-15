import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    environment: "node",
    include: ["**/*.test.ts", "**/*.test.tsx"],
    // e2e/ holds Playwright specs (`npm run test:e2e`), never run by Vitest.
    exclude: ["node_modules/**", ".next/**", "e2e/**"],
  },
});
