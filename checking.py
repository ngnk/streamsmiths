import polars as pl

df = pl.read_parquet("silver_data_v2/videos_v2.parquet")
df.write_csv("silver_data_v2/videos_v2.csv")

df = pl.read_parquet("silver_data_v2/channels_v2.parquet")
df.write_csv("silver_data_v2/channels_v2.csv")

df = pl.read_parquet("silver_data_v2/trending_videos_v2.parquet")
df.write_csv("silver_data_v2/trending_videos_v2.csv")
