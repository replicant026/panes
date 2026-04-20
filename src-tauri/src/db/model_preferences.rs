use anyhow::Context;
use rusqlite::params;

use crate::models::ModelPreferenceDto;

use super::Database;

pub fn list_for_workspace_user(
    db: &Database,
    workspace_id: &str,
    user_id: &str,
) -> anyhow::Result<Vec<ModelPreferenceDto>> {
    let conn = db.connect()?;
    let mut stmt = conn.prepare(
        "SELECT workspace_id, user_id, engine_id, model_id, is_favorite, is_enabled, updated_at
         FROM model_preferences
         WHERE workspace_id = ?1 AND user_id = ?2
         ORDER BY engine_id, model_id",
    )?;

    let rows = stmt.query_map(params![workspace_id, user_id], |row| {
        Ok(ModelPreferenceDto {
            workspace_id: row.get(0)?,
            user_id: row.get(1)?,
            engine_id: row.get(2)?,
            model_id: row.get(3)?,
            is_favorite: row.get::<_, i64>(4)? > 0,
            is_enabled: row.get::<_, i64>(5)? > 0,
            updated_at: row.get(6)?,
        })
    })?;

    let mut out = Vec::new();
    for row in rows {
        out.push(row?);
    }
    Ok(out)
}

pub fn upsert(
    db: &Database,
    workspace_id: &str,
    user_id: &str,
    engine_id: &str,
    model_id: &str,
    is_favorite: bool,
    is_enabled: bool,
) -> anyhow::Result<ModelPreferenceDto> {
    let conn = db.connect()?;
    conn.execute(
        "INSERT INTO model_preferences (
            workspace_id, user_id, engine_id, model_id, is_favorite, is_enabled, updated_at
         ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, datetime('now'))
         ON CONFLICT(workspace_id, user_id, engine_id, model_id) DO UPDATE SET
            is_favorite = excluded.is_favorite,
            is_enabled = excluded.is_enabled,
            updated_at = datetime('now')",
        params![
            workspace_id,
            user_id,
            engine_id,
            model_id,
            if is_favorite { 1 } else { 0 },
            if is_enabled { 1 } else { 0 }
        ],
    )
    .with_context(|| {
        format!(
            "failed to save model preference for {workspace_id}/{user_id}/{engine_id}/{model_id}"
        )
    })?;

    let mut rows = list_for_workspace_user(db, workspace_id, user_id)?
        .into_iter()
        .filter(|pref| pref.engine_id == engine_id && pref.model_id == model_id);
    rows.next()
        .ok_or_else(|| anyhow::anyhow!("saved model preference could not be reloaded"))
}
