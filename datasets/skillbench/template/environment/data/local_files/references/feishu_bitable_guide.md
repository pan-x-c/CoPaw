# 飞书多维表格集成指南

## 概述
飞书多维表格（Bitable）是一个强大的数据管理工具，支持通过 API 进行读写操作，适合用于医疗预约系统的号源管理和预约记录存储。

## 核心 API

### 获取表格数据
```
GET https://open.feishu.cn/open-apis/bitable/v1/apps/{app_token}/tables/{table_id}/records
```

### 新增记录
```
POST https://open.feishu.cn/open-apis/bitable/v1/apps/{app_token}/tables/{table_id}/records
Content-Type: application/json

{
  "fields": {
    "患者姓名": "张三",
    "联系电话": "13800138000",
    "预约科室": "内科",
    "预约医生": "张伟医生",
    "预约时间": "2026-04-20 10:00",
    "审批状态": "待审批"
  }
}
```

### 更新记录
```
PATCH https://open.feishu.cn/open-apis/bitable/v1/apps/{app_token}/tables/{table_id}/records/{record_id}
```

## 认证方式
使用 App Access Token 或 User Access Token，通过 Authorization 头传递：
```
Authorization: Bearer {access_token}
```

## 最佳实践
1. 批量操作时使用 batch 接口减少请求次数
2. 设置合理的重试机制处理限流
3. 使用字段 ID 而非字段名提高稳定性
4. 定期备份重要数据

## 常见问题
- Q: 如何获取 app_token？
  A: 在多维表格 URL 中，格式为 https://feishu.cn/base/{app_token}
  
- Q: 审批流如何触发？
  A: 通过飞书审批 API 创建审批实例，或使用自动化流程
