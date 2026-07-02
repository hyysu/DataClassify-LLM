"""Synthetic field metadata generation for Bayes classifier training.

The generator creates fake database-field metadata, not real personal data.
Generated sample values are used only as local format signals; the feature
extractor converts them into safe profiles before model training.
"""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Callable


SyntheticRecord = dict[str, object]
SampleFactory = Callable[[random.Random], str]


@dataclass(frozen=True)
class SyntheticCategorySpec:
    """Template pool for one data category."""

    label: str
    table_names: tuple[str, ...]
    table_comments: tuple[str, ...]
    column_names: tuple[str, ...]
    column_comments: tuple[str, ...]
    data_types: tuple[str, ...]
    business_contexts: tuple[str, ...]
    sample_factory: SampleFactory
    nullable: bool | None = None
    is_unique: bool | None = None


def _digits(rng: random.Random, length: int) -> str:
    return "".join(rng.choice("0123456789") for _ in range(length))


def _letters(rng: random.Random, length: int) -> str:
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    return "".join(rng.choice(alphabet) for _ in range(length))


def _fake_name(rng: random.Random) -> str:
    surnames = ("赵", "钱", "孙", "李", "周", "吴", "郑", "王")
    given = ("明", "华", "磊", "芳", "娜", "强", "敏", "洋")
    return rng.choice(surnames) + rng.choice(given)


def _fake_id_number(rng: random.Random) -> str:
    area = rng.choice(("110101", "310101", "440305", "320102"))
    year = rng.randint(1970, 2005)
    month = rng.randint(1, 12)
    day = rng.randint(1, 28)
    sequence = _digits(rng, 3)
    checksum = rng.choice("0123456789X")
    return f"{area}{year:04d}{month:02d}{day:02d}{sequence}{checksum}"


def _fake_phone(rng: random.Random) -> str:
    prefix = rng.choice(("138", "139", "150", "177", "186"))
    return prefix + _digits(rng, 8)


def _fake_email(rng: random.Random) -> str:
    return f"{_letters(rng, 6)}{rng.randint(10, 99)}@example.com"


def _fake_bank_card(rng: random.Random) -> str:
    return rng.choice(("622202", "621700", "622848")) + _digits(rng, rng.randint(10, 13))


def _fake_decimal(rng: random.Random) -> str:
    return f"{rng.randint(10, 999999)}.{rng.randint(0, 99):02d}"


def _fake_date(rng: random.Random) -> str:
    return f"{rng.randint(2020, 2026):04d}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"


def _fake_ip(rng: random.Random) -> str:
    return f"10.{rng.randint(0, 255)}.{rng.randint(0, 255)}.{rng.randint(1, 254)}"


def _fake_url(rng: random.Random) -> str:
    return f"https://example.com/{_letters(rng, 5)}/{rng.randint(100, 999)}"


def _fake_social_credit_code(rng: random.Random) -> str:
    alphabet = "0123456789ABCDEFGHJKLMNPQRTUWXY"
    return "".join(rng.choice(alphabet) for _ in range(18))


def _fake_health_value(rng: random.Random) -> str:
    return rng.choice(("120/80", "心率72", "血糖5.6", "过敏性鼻炎", "体温36.6"))


def _fake_token(rng: random.Random) -> str:
    return rng.choice(("token_", "sk-", "session_")) + _letters(rng, 12) + _digits(rng, 4)


def _fake_plain_text(rng: random.Random) -> str:
    return rng.choice(("正常", "启用", "待处理", "公开公告", "测试备注", "默认配置"))


SYNTHETIC_CATEGORY_SPECS: tuple[SyntheticCategorySpec, ...] = (
    SyntheticCategorySpec(
        label="个人基本信息",
        table_names=("user_profile", "member_basic", "customer_profile"),
        table_comments=("用户资料表", "会员基础信息表", "客户档案表"),
        column_names=("real_name", "user_name", "gender", "birthday", "age"),
        column_comments=("姓名", "用户姓名", "性别", "出生日期", "年龄"),
        data_types=("varchar", "date", "int"),
        business_contexts=("用户管理", "会员服务", "客户档案"),
        sample_factory=lambda rng: rng.choice((_fake_name(rng), _fake_date(rng), str(rng.randint(18, 80)))),
        nullable=False,
    ),
    SyntheticCategorySpec(
        label="个人身份信息",
        table_names=("user_auth", "realname_verify", "kyc_profile"),
        table_comments=("实名认证表", "身份核验表", "客户实名信息表"),
        column_names=("id_card_no", "cert_no", "identity_number", "passport_no"),
        column_comments=("身份证号码", "证件号码", "身份标识号码", "护照号码"),
        data_types=("varchar", "char"),
        business_contexts=("实名认证", "身份核验", "开户校验"),
        sample_factory=_fake_id_number,
        nullable=False,
        is_unique=True,
    ),
    SyntheticCategorySpec(
        label="个人联系方式",
        table_names=("contact_book", "user_contact", "customer_contact"),
        table_comments=("联系人表", "用户联系方式表", "客户联系表"),
        column_names=("mobile_phone", "phone_number", "email_address", "receiver_tel"),
        column_comments=("手机号", "联系电话", "电子邮箱", "收件人电话"),
        data_types=("varchar", "char"),
        business_contexts=("联系通知", "用户资料管理", "订单履约"),
        sample_factory=lambda rng: rng.choice((_fake_phone(rng), _fake_email(rng))),
        nullable=False,
        is_unique=True,
    ),
    SyntheticCategorySpec(
        label="个人财产信息",
        table_names=("payment_account", "bank_binding", "wallet_account"),
        table_comments=("支付账户表", "银行卡绑定表", "钱包账户表"),
        column_names=("bank_card_no", "pay_account_id", "account_balance", "credit_amount"),
        column_comments=("银行卡号", "支付账户", "账户余额", "授信金额"),
        data_types=("varchar", "decimal"),
        business_contexts=("支付结算", "钱包管理", "金融账户"),
        sample_factory=lambda rng: rng.choice((_fake_bank_card(rng), _fake_decimal(rng))),
        nullable=False,
    ),
    SyntheticCategorySpec(
        label="个人健康生理信息",
        table_names=("medical_record", "health_archive", "physical_index"),
        table_comments=("病历记录表", "健康档案表", "体征指标表"),
        column_names=("diagnosis_result", "blood_pressure", "heart_rate", "medical_history"),
        column_comments=("诊断结果", "血压指标", "心率", "病史记录"),
        data_types=("varchar", "text"),
        business_contexts=("互联网医疗", "健康档案", "问诊服务"),
        sample_factory=_fake_health_value,
        nullable=True,
    ),
    SyntheticCategorySpec(
        label="个人网络身份标识信息",
        table_names=("login_account", "device_fingerprint", "oauth_identity"),
        table_comments=("登录账户表", "设备指纹表", "网络身份表"),
        column_names=("login_id", "device_id", "client_ip", "session_token", "open_id"),
        column_comments=("登录账号", "设备唯一标识", "客户端IP", "会话令牌", "第三方开放ID"),
        data_types=("varchar", "text"),
        business_contexts=("登录认证", "终端风控", "开放平台"),
        sample_factory=lambda rng: rng.choice((_fake_ip(rng), _fake_token(rng), f"openid_{_letters(rng, 10)}")),
        nullable=False,
        is_unique=True,
    ),
    SyntheticCategorySpec(
        label="组织经营信息",
        table_names=("finance_report", "crm_customer", "business_metric"),
        table_comments=("财务经营报表", "客户商业信息表", "经营指标表"),
        column_names=("revenue_amount", "profit_amount", "customer_level", "budget_amount"),
        column_comments=("收入金额", "利润金额", "客户分层等级", "预算金额"),
        data_types=("decimal", "varchar"),
        business_contexts=("经营分析", "财务报表", "CRM管理"),
        sample_factory=lambda rng: rng.choice((_fake_decimal(rng), rng.choice(("A类客户", "重点客户", "战略客户")))),
        nullable=False,
    ),
    SyntheticCategorySpec(
        label="业务交易信息",
        table_names=("order_main", "payment_order", "trade_flow"),
        table_comments=("订单主表", "支付订单表", "交易流水表"),
        column_names=("order_no", "trade_no", "total_amount", "payment_time", "refund_no"),
        column_comments=("订单编号", "交易流水号", "订单总金额", "支付时间", "退款单号"),
        data_types=("varchar", "decimal", "datetime"),
        business_contexts=("电商订单", "支付结算", "交易履约"),
        sample_factory=lambda rng: rng.choice((f"ORD{rng.randint(20200101, 20261231)}{_digits(rng, 6)}", _fake_decimal(rng), _fake_date(rng))),
        nullable=False,
    ),
    SyntheticCategorySpec(
        label="系统运行日志",
        table_names=("access_log", "login_log", "system_event"),
        table_comments=("访问日志表", "登录日志表", "系统事件表"),
        column_names=("remote_ip", "request_url", "status_code", "event_time", "user_agent"),
        column_comments=("访问来源IP", "请求地址", "状态码", "事件时间", "浏览器标识"),
        data_types=("varchar", "int", "datetime", "text"),
        business_contexts=("系统运维", "安全审计", "访问日志"),
        sample_factory=lambda rng: rng.choice((_fake_ip(rng), _fake_url(rng), str(rng.choice((200, 401, 403, 500))))),
        nullable=False,
    ),
    SyntheticCategorySpec(
        label="公共或普通业务数据",
        table_names=("notice_board", "product_catalog", "help_article"),
        table_comments=("公告信息表", "商品目录表", "帮助文章表"),
        column_names=("public_title", "product_name", "article_title", "display_order"),
        column_comments=("公开公告标题", "商品名称", "文章标题", "展示排序"),
        data_types=("varchar", "int", "text"),
        business_contexts=("门户网站", "商品展示", "公开内容管理"),
        sample_factory=_fake_plain_text,
        nullable=True,
    ),
    SyntheticCategorySpec(
        label="其他",
        table_names=("misc_table", "temp_record", "migration_task"),
        table_comments=("杂项记录表", "临时记录表", "迁移任务表"),
        column_names=("remark", "extra_info", "reserved_field", "unknown_value"),
        column_comments=("备注信息", "扩展信息", "预留字段", "未知值"),
        data_types=("varchar", "text"),
        business_contexts=("临时处理", "数据迁移", "无法明确归类"),
        sample_factory=_fake_plain_text,
        nullable=True,
    ),
)


def _sample_values(factory: SampleFactory, rng: random.Random, sample_count: int) -> tuple[str, ...]:
    return tuple(factory(rng) for _ in range(sample_count))


def generate_synthetic_training_records(
    records_per_category: int = 40,
    seed: int = 42,
    sample_values_per_record: int = 3,
) -> list[SyntheticRecord]:
    """Generate labeled field records for Bayes classifier training.

    The result is deterministic for a given seed. It is intended to be passed
    directly into ``DataFieldBayesClassifier.fit`` and does not write a dataset
    file by default.
    """

    if records_per_category <= 0:
        raise ValueError("records_per_category must be positive")
    if sample_values_per_record <= 0:
        raise ValueError("sample_values_per_record must be positive")

    rng = random.Random(seed)
    records: list[SyntheticRecord] = []
    for spec in SYNTHETIC_CATEGORY_SPECS:
        for _ in range(records_per_category):
            record = {
                "table_name": rng.choice(spec.table_names),
                "table_comment": rng.choice(spec.table_comments),
                "column_name": rng.choice(spec.column_names),
                "column_comment": rng.choice(spec.column_comments),
                "data_type": rng.choice(spec.data_types),
                "business_context": rng.choice(spec.business_contexts),
                "sample_values": _sample_values(spec.sample_factory, rng, sample_values_per_record),
                "nullable": spec.nullable if spec.nullable is not None else rng.choice((True, False)),
                "is_primary_key": False,
                "is_unique": spec.is_unique if spec.is_unique is not None else rng.choice((False, False, True)),
                "label": spec.label,
            }
            records.append(record)

    rng.shuffle(records)
    return records


def synthetic_label_set() -> set[str]:
    """Return all labels covered by the synthetic generator."""

    return {spec.label for spec in SYNTHETIC_CATEGORY_SPECS}

