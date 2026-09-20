"""ShopCare 中间件包 (backend/middleware)

目前只有一个 request_context 中间件(请求 ID / 限流 / 访问日志).
鉴权没有放在这里: 它是"按接口不同而不同"的(有的免登录, 有的要管理员),
放在 middleware 里就得维护一张路径白名单, 不如用 FastAPI 的依赖注入显式声明.
"""