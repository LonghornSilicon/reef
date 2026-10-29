# Atlas

Atlas is an evidence-linked engineering workspace for the Coral SoC. It combines the existing interactive architecture model with a Git-backed wiki, grounded AI explanations, native task management, and a versioned document library.

## Local development

Requirements: Node.js 22+, npm, and Python 3.

```bash
cp .env.example .env.local
npm install --legacy-peer-deps
npm run dev
```

Without external services, development mode uses a local demo administrator, deterministic evidence search over the checked-in wiki, in-memory conversations/tasks, and temporary document storage. This makes every product surface reviewable without credentials.

## Production configuration

1. Set `DATABASE_URL` and run `npm run db:migrate`.
2. Configure Google or Microsoft Entra ID OAuth plus `AUTH_SECRET` and `ATLAS_ALLOWED_EMAIL_DOMAINS`.
3. Configure an S3-compatible bucket for managed documents.
4. Set `OPENAI_API_KEY`, `OPENAI_CHAT_MODEL`, and `OPENAI_IMAGE_MODEL`.
5. Build and start the container with `npm run build` and `npm start`, or use the included `Dockerfile`.
6. As an Atlas administrator, open **Admin** and index the current Git revision. CI may call `POST /api/admin/ingestion` with `X-Atlas-Ingestion-Secret`.

Provider model identifiers are environment configuration rather than application constants. Generated images are returned to the requesting browser and are not stored by Atlas. Personal conversation threads expire after 30 days.

## Canonical sources

- `architecture/soc.yaml` is the architecture model.
- `docs/` is the editable wiki.
- `site/` remains the source for the embedded architecture viewer and read-only VS Code companion.
- `db/schema.sql` defines collaborative application data.

Run the complete verification suite with:

```bash
npm run check
```
