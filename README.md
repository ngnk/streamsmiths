---
# YouTube Pipeline V1
---

![V1](https://github.com/ngnk/streamsmiths/actions/workflows/pipeline.yml/badge.svg)

---

## Database Schema

### Channels

| Column             | Data Type | Example                     |
| ------------------ | --------- | --------------------------- |
| `channel_id`       | `TEXT`    | `UCANLZYMidaCbLQFWXBC95Jg`  |
| `channel_title`    | `TEXT`    | `TaylorSwiftVEVO`           |
| `description`      | `TEXT`    | `I'm the problem, it's me.` |
| `country`          | `TEXT`    | `US`                        |
| `published_at`     | `TEXT`    | `2009-05-12T05:29:33Z`      |
| `subscriber_count` | `INTEGER` | `25300000`                  |
| `view_count`       | `INTEGER` | `0`                         |
| `video_count`      | `INTEGER` | `75`                        |
| `ingest_timestamp` | `TEXT`    | `2025-11-15 21:52:27`       |


### videos_log

| Column             | Data Type              | Example                       |
| ------------------ | ---------------------- | ----------------------------- |
| `video_id`         | TEXT                   | `oWVYzCPs3nE`                 |
| `channel_id`       | TEXT                   | `UCANLZYMidaCbLQFWXBC95Jg`    |
| `video_title`      | TEXT                   | `TaylorSwiftVEVO Live Stream` |
| `description`      | TEXT (nullable)        | `NULL` / empty                |
| `published_at`     | TIMESTAMP (ISO string) | `2017-12-01T21:00:48Z`        |
| `view_count`       | BIGINT                 | `0`                           |
| `like_count`       | BIGINT                 | `63`                          |
| `comment_count`    | BIGINT                 | `1`                           |
| `duration`         | TEXT (ISO 8601)        | `P0D`                         |
| `ingest_timestamp` | TIMESTAMP (ISO string) | `2025-11-15T02:13:05.493325`  |

---
