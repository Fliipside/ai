-- Schema `memory`: conversation memory search (chunks + vectors) and its source data.
-- Safe to re-run (IF NOT EXISTS). To empty the data but keep the tables, use 03_clear.sql.
-- Agree on the final design with the backend before moving it into migrations.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE SCHEMA IF NOT EXISTS memory;
SET search_path = memory, public;

-- Source data (ids are the fixture ids, e.g. S001-I2-U08)
CREATE TABLE IF NOT EXISTS app_users (
    user_id int PRIMARY KEY,
    team_id int NOT NULL
);

CREATE TABLE IF NOT EXISTS companies (
    id       text PRIMARY KEY,
    name     text NOT NULL,
    industry text
);

CREATE TABLE IF NOT EXISTS contacts (
    id           text PRIMARY KEY,
    company_id   text NOT NULL REFERENCES companies,
    name         text NOT NULL,
    title        text,
    role_in_deal text
);

CREATE TABLE IF NOT EXISTS interactions (
    id            text PRIMARY KEY,
    company_id    text NOT NULL REFERENCES companies,
    owner_user_id int NOT NULL REFERENCES app_users,
    visibility    text NOT NULL CHECK (visibility IN ('private', 'team')),
    kind          text NOT NULL,
    occurred_at   timestamptz NOT NULL,
    title         text
);

CREATE TABLE IF NOT EXISTS utterances (
    id             text PRIMARY KEY,
    interaction_id text NOT NULL REFERENCES interactions ON DELETE CASCADE,
    seq            int NOT NULL,
    start_sec      real NOT NULL,
    end_sec        real NOT NULL,
    speaker        text NOT NULL,
    role           text NOT NULL,
    contact_id     text REFERENCES contacts,
    text           text NOT NULL
);

CREATE TABLE IF NOT EXISTS facts (
    id                   text PRIMARY KEY,
    interaction_id       text NOT NULL REFERENCES interactions ON DELETE CASCADE,
    type                 text NOT NULL,
    said_by              text NOT NULL,
    content              text NOT NULL,
    source_utterance_ids text[] NOT NULL,
    owner                text,
    due_at               date,
    due_text             text,
    status               text,
    supersedes           text
);

-- Search data. Permission columns are copied so search needs no joins.
CREATE TABLE IF NOT EXISTS memory_chunks (
    id              bigserial PRIMARY KEY,
    company_id      text NOT NULL REFERENCES companies,
    interaction_id  text NOT NULL REFERENCES interactions ON DELETE CASCADE,
    owner_user_id   int NOT NULL,
    team_id         int NOT NULL,
    visibility      text NOT NULL,
    occurred_at     timestamptz NOT NULL,
    content         text NOT NULL,              -- shown to the user
    embed_text      text NOT NULL,              -- content + company/date/title header
    utterance_ids   text[] NOT NULL,            -- evidence, for original speech and time
    start_sec       real NOT NULL,
    end_sec         real NOT NULL,
    embedding       vector(1024) NOT NULL,
    embedding_model text NOT NULL
);

CREATE INDEX IF NOT EXISTS memory_chunks_embedding_idx
    ON memory_chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS memory_chunks_company_idx
    ON memory_chunks (company_id, occurred_at DESC);
