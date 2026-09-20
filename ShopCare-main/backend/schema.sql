-- ============================================================================
-- ShopCare 建表脚本 (backend/schema.sql)
--
-- 使用方式(两种都行):
--   1) 手动: mysql -u root -p < backend/schema.sql
--   2) 自动: 后端启动时若 DB_AUTO_CREATE=true 会自动执行本文件(需要账号有建库权限)
--
-- 字符集统一 utf8mb4: 工单文本里 emoji、生僻字很常见, utf8(3 字节)会直接插入失败.
-- ============================================================================

CREATE DATABASE IF NOT EXISTS `shopcare`
    DEFAULT CHARACTER SET utf8mb4
    DEFAULT COLLATE utf8mb4_unicode_ci;

USE `shopcare`;

-- ---------------------------------------------------------------------------
-- 用户表: 客服/主管/管理员
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS `users` (
    `id`            BIGINT       NOT NULL AUTO_INCREMENT,
    `username`      VARCHAR(64)  NOT NULL COMMENT '登录名',
    `password_hash` VARCHAR(255) NOT NULL COMMENT 'pbkdf2_sha256$迭代$盐$哈希',
    `role`          VARCHAR(16)  NOT NULL DEFAULT 'user' COMMENT 'admin / user',
    `created_at`    DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (`id`),
    UNIQUE KEY `uk_username` (`username`)
) ENGINE = InnoDB DEFAULT CHARSET = utf8mb4 COMMENT '用户';

-- ---------------------------------------------------------------------------
-- 工单表: 一次分类请求的完整快照
--   labels / confidences 存 JSON 字符串(str), 不用 MySQL 原生 JSON 类型,
--   这样同一套 DAO 代码在 MySQL 5.7 以下也能跑, 兼容性更好.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS `tickets` (
    `id`              BIGINT       NOT NULL AUTO_INCREMENT,
    `ticket_id`       VARCHAR(32)  NOT NULL COMMENT '业务工单号 SC20260920-1234',
    `text`            TEXT         NOT NULL COMMENT '原始工单文本',
    `text_masked`     TEXT                  COMMENT '脱敏后的文本(手机号/单号打码)',
    `model_used`      VARCHAR(24)           COMMENT 'rf-tfidf / fasttext / bert',
    `labels`          TEXT                  COMMENT '激活标签 JSON 数组',
    `confidences`     TEXT                  COMMENT '各标签置信度 JSON 对象',
    `avg_confidence`  FLOAT                 COMMENT '激活标签平均置信度',
    `rejected`        TINYINT(1)   NOT NULL DEFAULT 0 COMMENT '是否被拒识',
    `reject_reason`   VARCHAR(32)           COMMENT '拒识原因码',
    `sentiment`       VARCHAR(16)           COMMENT 'positive/neutral/negative',
    `priority`        VARCHAR(4)            COMMENT 'P0/P1/P2',
    `dept`            VARCHAR(32)           COMMENT '建议处理部门',
    `sla_hours`       INT                   COMMENT '要求响应时长',
    `suggested_reply` TEXT                  COMMENT '推荐回复草稿',
    `needs_approval`  TINYINT(1)   NOT NULL DEFAULT 0 COMMENT '回复是否需要人工确认',
    `resolved_by`     VARCHAR(8)            COMMENT 'model / llm / human / none',
    `latency_ms`      FLOAT                 COMMENT '端到端耗时',
    `status`          VARCHAR(16)  NOT NULL DEFAULT 'pending' COMMENT 'pending/reviewed/closed',
    `created_at`      DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (`id`),
    UNIQUE KEY `uk_ticket_id` (`ticket_id`),
    -- 查询模式主要是"按时间倒序翻页"和"按状态/优先级过滤", 所以建这几个索引
    KEY `idx_created` (`created_at`),
    KEY `idx_status` (`status`),
    KEY `idx_priority` (`priority`),
    KEY `idx_dept` (`dept`)
) ENGINE = InnoDB DEFAULT CHARSET = utf8mb4 COMMENT '客服工单';

-- ---------------------------------------------------------------------------
-- 人工复核表: 记录每一次"人对模型结果做了什么"
--   这张表是**模型迭代的燃料**: 修正后的标签就是下一轮训练的增量样本.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS `reviews` (
    `id`               BIGINT      NOT NULL AUTO_INCREMENT,
    `ticket_id`        VARCHAR(32) NOT NULL,
    `action`           VARCHAR(16) NOT NULL COMMENT 'approve / correct / reject',
    `corrected_labels` TEXT                 COMMENT '人工修正后的标签 JSON',
    `note`             VARCHAR(500)         COMMENT '复核备注',
    `reviewer`         VARCHAR(64)          COMMENT '复核人',
    `created_at`       DATETIME    NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (`id`),
    KEY `idx_ticket` (`ticket_id`),
    KEY `idx_reviewer` (`reviewer`)
) ENGINE = InnoDB DEFAULT CHARSET = utf8mb4 COMMENT '人工复核记录';

-- ---------------------------------------------------------------------------
-- 审计日志: 谁在什么时候对哪条工单做了什么
--   合规场景(尤其是涉及退款)基本都会要求这张表; 写入失败绝不影响主流程.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS `audit_logs` (
    `id`         BIGINT       NOT NULL AUTO_INCREMENT,
    `username`   VARCHAR(64)           COMMENT '操作人',
    `action`     VARCHAR(64)  NOT NULL COMMENT '动作: classify/login/review/...',
    `target`     VARCHAR(64)           COMMENT '操作对象(通常是 ticket_id)',
    `detail`     TEXT                  COMMENT '细节 JSON',
    `created_at` DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (`id`),
    KEY `idx_username` (`username`),
    KEY `idx_created` (`created_at`)
) ENGINE = InnoDB DEFAULT CHARSET = utf8mb4 COMMENT '审计日志';