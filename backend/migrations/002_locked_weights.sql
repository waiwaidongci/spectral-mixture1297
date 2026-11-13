-- Locked weighing support for existing databases (new databases bootstrap from metadata).
-- Legacy runs keep locked_weights = NULL / [] and lock_basis = '' and are read as unlocked.
ALTER TABLE unmix_runs ADD COLUMN locked_weights JSON NULL;
ALTER TABLE unmix_runs ADD COLUMN lock_basis TEXT NULL;
