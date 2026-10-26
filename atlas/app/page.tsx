import { redirect } from "next/navigation";

export default async function Home({ searchParams }: { searchParams: Promise<Record<string, string>> }) {
  const q = new URLSearchParams(await searchParams).toString();
  redirect(`/canvas${q ? `?${q}` : ""}`);
}
