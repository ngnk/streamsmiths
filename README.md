---
# YouTube Pipeline V2
---

![Pipeline V1](https://github.com/ngnk/streamsmiths/actions/workflows/pipeline.yml/badge.svg)

![Pipeline V2](https://github.com/ngnk/streamsmiths/actions/workflows/youtube_pipeline_v2.yml/badge.svg)

---

## Channels

1. **Cocomelon**
2. **LikeNastya**
3. **Ryan's World**
4. **MrBeast**
5. **PewDiePie**
6. **Sidemen**
7. **TEDx Talks**
8. **Kurzgesagt**
9. **Nick DiGiovanni**
10. **CNN**
11. **Yuya**
12. **James Charles**
13. **JuegaGerman**
14. **Markiplier**
15. **Taylor Swift**
16. **Michael Jackson**
17. **BLACKPINK**
18. **Bad Bunny**
19. **T-Series**
20. **UR Cristiano**
21. **Tibo InShape**
22. **Chloe Ting**
23. **Dude Perfect**
24. **Linus Tech Tips**

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


### channels_log_v2

| Column                | Data Type | Example                                    |
| --------------------- | --------- | ------------------------------------------ |
| `channel_id`          | `TEXT`    | `UCANLZYMidaCbLQFWXBC95Jg`                 |
| `channel_title`       | `TEXT`    | `TaylorSwiftVEVO`                          |
| `custom_url`          | `TEXT`    | `@TaylorSwiftVEVO`                         |
| `description`         | `TEXT`    | `I'm the problem, it's me.`                |
| `country`             | `TEXT`    | `US`                                       |
| `published_at`        | `TEXT`    | `2009-05-12T05:29:33Z`                     |
| `thumbnail_url`       | `TEXT`    | `https://yt3.ggpht.com/wiJf-C4mG_oU5H...`  |
| `subscriber_count`    | `INTEGER` | `25300000`                                 |
| `view_count`          | `INTEGER` | `0`                                        |
| `video_count`         | `INTEGER` | `75`                                       |
| `keywords`            | `TEXT`    | `Taylor Swift,Music,Pop,Official,Video...` |
| `topic_categories`    | `TEXT`    | `https://en.wikipedia.org/wiki/Music,...`  |
| `ingestion_timestamp` | `TEXT`    | `2025-11-15 21:52:27`                      |


### videos_log_v2

| Column                | Data Type          | Example                                                             |
| --------------------- | ------------------ | ------------------------------------------------------------------- |
| `video_id`            | TEXT               | `DMD2uthghWE`                                                       |
| `channel_id`          | TEXT               | `UCq-Fj5jknLsUf-MWSy4_brA`                                          |
| `video_title`         | TEXT               | `VARANASI to the WORLD - Mahesh Babu \| Priyanka Chopra \| ...`     |
| `description`         | TEXT               | `Presenting Mahesh Babu as Rudhra in #Varanasi 🔱\n\nOn the ris...` |
| `published_at`        | TIMESTAMP (string) | `2025-11-15 17:15:27`                                               |
| `thumbnail_url`       | TEXT               | `https://i.ytimg.com/vi/DMD2uthghWE/hqdefault.jpg`                  |
| `category_id`         | INT                | `10`                                                                |
| `tags`                | TEXT               | `tseries,tseries songs,Varanasi to world,Mahesh babu event,...`     |
| `view_count`          | BIGINT             | `2319545`                                                           |
| `like_count`          | BIGINT             | `184421`                                                            |
| `comment_count`       | BIGINT             | `8080`                                                              |
| `duration_seconds`    | INT                | `221`                                                               |
| `duration_iso`        | TEXT (ISO 8601)    | `PT3M41S`                                                           |
| `definition`          | TEXT               | `hd`                                                                |
| `caption`             | BOOLEAN            | `False`                                                             |
| `topic_categories`    | TEXT               | `https://en.wikipedia.org/wiki/Entertainment,https://en.wiki...`    |
| `engagement_rate`     | FLOAT              | `8.2991`                                                            |
| `views_per_day`       | BIGINT             | `2319545`                                                           |
| `days_since_publish`  | INT                | `1`                                                                 |
| `ingestion_timestamp` | TIMESTAMP (string) | `2025-11-15 22:04:26`                                               |


### trending_videos_log_v2

| Column                | Data Type          | Example                                                            |
| --------------------- | ------------------ | ------------------------------------------------------------------ |
| `video_id`            | TEXT               | `lLFoLJIXayk`                                                      |
| `channel_id`          | TEXT               | `UCI3H1FsjbdqGcLq93ZilV5g`                                         |
| `video_title`         | TEXT               | `NF - FEAR`                                                        |
| `description`         | TEXT               | `Official music video for “FEAR” by NF from the FEAR EP out no...` |
| `published_at`        | TIMESTAMP (string) | `2025-11-14 05:00:07`                                              |
| `thumbnail_url`       | TEXT               | `https://i.ytimg.com/vi/lLFoLJIXayk/hqdefault.jpg`                 |
| `category_id`         | INT                | `10`                                                               |
| `tags`                | TEXT               | `NF songs,nfrealmusic,nf youtube,nf lyrics,fear,nf fear,...`       |
| `view_count`          | BIGINT             | `1778275`                                                          |
| `like_count`          | BIGINT             | `157777`                                                           |
| `comment_count`       | BIGINT             | `13819`                                                            |
| `duration_seconds`    | INT                | `271`                                                              |
| `duration_iso`        | TEXT (ISO 8601)    | `PT4M31S`                                                          |
| `definition`          | TEXT               | `hd`                                                               |
| `caption`             | BOOLEAN            | `True`                                                             |
| `topic_categories`    | TEXT               | `https://en.wikipedia.org/wiki/Hip_hop_music,https://en.wiki...`   |
| `engagement_rate`     | FLOAT              | `9.6496`                                                           |
| `views_per_day`       | BIGINT             | `1778275`                                                          |
| `days_since_publish`  | INT                | `1`                                                                |
| `ingestion_timestamp` | TIMESTAMP (string) | `2025-11-15 22:04:27`                                              |
| `trending_region`     | TEXT               | `US`                                                               |

Note: engagement_rate = ((like_count + comment_count) / view_count) × 100

---
