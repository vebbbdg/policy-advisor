"""政策语料源清单（阶段 1.1）

仅收录官方来源，低频抓取公开页面，遵守 robots.txt 与站点条款。
每个源携带后续检索需要的元数据：政策主题 + 来源机构。
新增语料源时在 SOURCES 追加一条即可，无需改代码。
"""

SOURCES = [
    # ---------- USCIS：OPT / STEM OPT / 学生就业总览 ----------
    {
        "slug": "uscis-opt",
        "url": "https://www.uscis.gov/working-in-the-united-states/students-and-exchange-visitors/optional-practical-training-opt-for-f-1-students",
        "policy_topic": "OPT",
        "source_org": "USCIS",
    },
    {
        "slug": "uscis-stem-opt",
        "url": "https://www.uscis.gov/working-in-the-united-states/students-and-exchange-visitors/optional-practical-training-extension-for-stem-students-stem-opt",
        "policy_topic": "STEM_OPT",
        "source_org": "USCIS",
    },
    {
        "slug": "uscis-students-employment",
        "url": "https://www.uscis.gov/working-in-the-united-states/students-and-exchange-visitors/students-and-employment",
        "policy_topic": "GENERAL",
        "source_org": "USCIS",
    },
    {
        "slug": "uscis-policy-manual-f5",
        "url": "https://www.uscis.gov/policy-manual/volume-2-part-f-chapter-5",
        "policy_topic": "OPT",
        "source_org": "USCIS",
    },
    {
        "slug": "uscis-i765",
        "url": "https://www.uscis.gov/i-765",
        "policy_topic": "OPT",
        "source_org": "USCIS",
    },
    # ---------- DHS Study in the States（SEVP 官方）：主列表页只有导航，取实际内容页 ----------
    {
        "slug": "sit-student-employment-overview",
        "url": "https://studyinthestates.dhs.gov/sevis-help-hub/student-records/fm-student-employment/student-employment-overview",
        "policy_topic": "GENERAL",
        "source_org": "DHS-SEVP",
    },
    {
        "slug": "sit-cpt",
        "url": "https://studyinthestates.dhs.gov/sevis-help-hub/student-records/fm-student-employment/f-1-curricular-practical-training-cpt",
        "policy_topic": "CPT",
        "source_org": "DHS-SEVP",
    },
    {
        "slug": "sit-opt",
        "url": "https://studyinthestates.dhs.gov/sevis-help-hub/student-records/fm-student-employment/f-1-optional-practical-training-opt",
        "policy_topic": "OPT",
        "source_org": "DHS-SEVP",
    },
    {
        "slug": "sit-stem-opt",
        "url": "https://studyinthestates.dhs.gov/sevis-help-hub/student-records/fm-student-employment/f-1-stem-optional-practical-training-opt",
        "policy_topic": "STEM_OPT",
        "source_org": "DHS-SEVP",
    },
]
