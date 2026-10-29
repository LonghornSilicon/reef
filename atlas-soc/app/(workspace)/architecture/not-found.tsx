import Link from "next/link";
import { Icon } from "@/components/icons";

export default function ArchitectureNotFound() {
  return (
    <div className="surface-notfound">
      <span className="inspector-eyebrow">Unknown block</span>
      <h1>No such block ID</h1>
      <p>
        That identifier is not in <code>architecture/soc.yaml</code>. Block IDs are stable, so a broken link usually means
        the block was renamed in Git.
      </p>
      <Link className="action-primary" href="/architecture">
        <Icon name="architecture" aria-hidden="true" />
        <span>
          <strong>Open the explorer</strong>
          <small>Browse or filter all blocks</small>
        </span>
        <Icon name="arrow" aria-hidden="true" />
      </Link>
    </div>
  );
}
