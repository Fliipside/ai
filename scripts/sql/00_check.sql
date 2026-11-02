-- Run first in DBeaver. Read-only: shows what the server already has.

-- PostgreSQL version
SELECT version();

-- Installed extensions. Need `vector`; 0.8+ also allows filtered HNSW search (iterative scan).
SELECT extname, extversion FROM pg_extension ORDER BY extname;

-- Extensions that can be installed on this server
SELECT name, default_version, installed_version
FROM pg_available_extensions
WHERE name IN ('vector', 'pg_trgm');

-- Existing schemas and tables (so the sandbox does not collide with anything)
SELECT table_schema, table_name
FROM information_schema.tables
WHERE table_schema NOT IN ('pg_catalog', 'information_schema')
ORDER BY table_schema, table_name;

-- Current user and whether it may create extensions/schemas
SELECT current_user, current_database(),
       has_database_privilege(current_database(), 'CREATE') AS can_create_schema;
