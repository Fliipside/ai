-- Empty every table in schema memory but keep the tables, indexes and schema.
-- Only schema memory is touched. To remove just the dummy data, re-run
-- scripts/load_memory_fixtures.py instead (it replaces fixture rows only).
TRUNCATE memory.memory_chunks, memory.facts, memory.utterances, memory.interactions,
         memory.contacts, memory.companies, memory.app_users;
