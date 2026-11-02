-- Run after scripts/load_memory_sandbox.py has filled the sandbox.
SET search_path = memory_sandbox, public;

-- Row counts
SELECT 'companies' AS t, count(*) FROM companies
UNION ALL SELECT 'interactions', count(*) FROM interactions
UNION ALL SELECT 'utterances', count(*) FROM utterances
UNION ALL SELECT 'facts', count(*) FROM facts
UNION ALL SELECT 'memory_chunks', count(*) FROM memory_chunks;

-- Chunks per company
SELECT c.name, count(*) AS chunks
FROM memory_chunks m JOIN companies c ON c.id = m.company_id
GROUP BY c.name ORDER BY chunks DESC;

-- Timeline of one customer
SELECT i.occurred_at, i.kind, i.title, count(u.id) AS utterances
FROM interactions i JOIN utterances u ON u.interaction_id = i.id
WHERE i.company_id = 'S001-CO1'
GROUP BY i.id ORDER BY i.occurred_at;

-- Open promises (structured question: no vectors needed)
SELECT c.name, f.owner, f.due_at, f.due_text, f.content
FROM facts f
JOIN interactions i ON i.id = f.interaction_id
JOIN companies c ON c.id = i.company_id
WHERE f.type = 'promise' AND f.status = 'open'
ORDER BY f.due_at NULLS LAST;

-- Vector search by hand: find chunks close to an existing chunk.
-- (A text question must be embedded in Python first; see retrieval_eval.py --backend pg.)
SELECT m.id, left(m.content, 60) AS content,
       1 - (m.embedding <=> (SELECT embedding FROM memory_chunks WHERE id = 1)) AS similarity
FROM memory_chunks m
WHERE m.company_id = (SELECT company_id FROM memory_chunks WHERE id = 1)
ORDER BY m.embedding <=> (SELECT embedding FROM memory_chunks WHERE id = 1)
LIMIT 5;
