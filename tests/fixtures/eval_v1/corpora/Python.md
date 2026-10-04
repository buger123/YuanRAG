# Python

基础数据结构与语法 函数基础 异常处理 

迭代器与生成器 装饰器 面向对象编程 上下文管理器 模块与包管理 

代码规范与类型提示 环境与依赖管理 日志系统 异常处理最佳实践 单元测试

核心标准库 通用必备第三方库

并发基础 异步编程

性能优化 内存管理 设计模式 部署与运维 安全

### 一、核心基础（所有方向通用，必须熟练）

这部分是写代码的根基，要求做到高频用法不假思索

1. 基础数据结构与语法
   - 核心类型：`list / dict / tuple / set` 的特性、常用方法与适用场景，重点掌握字典底层逻辑、列表 / 字典 / 集合推导式
   - 变量与作用域：LEGB 规则，全局 / 局部变量的边界
   - 流程控制：条件判断、循环、`break/continue/else` 的完整用法
2. 函数基础
   - 参数体系：位置参数、关键字参数、默认参数、可变参数 `*args` / `**kwargs`
   - 函数一等公民特性、匿名函数 `lambda`
3. 基础异常处理
   - `try-except-else-finally` 结构，核心原则：禁止裸 `except`，只捕获预期异常

### 二、进阶语法特性（中高级分水岭）

这部分是区分「会写 Python」和「写生产级 Python」的关键

1. 迭代器与生成器
   - 迭代器协议、`yield` 关键字、生成器表达式
   - 惰性求值原理，大文件 / 大数据流处理的最佳实践
2. 装饰器
   - 闭包原理，无参 / 带参装饰器、类装饰器写法
   - 典型生产应用：日志埋点、性能计时、权限校验、结果缓存
3. 面向对象编程（OOP）
   - 类与实例、封装 / 继承 / 多态，MRO 继承顺序
   - 核心魔法方法：`__init__` / `__new__` / `__str__` 等
   - 实例方法 / 类方法 / 静态方法的区别与场景
   - 单例、工厂等常用设计模式的 Pythonic 实现
4. 上下文管理器
   - `with` 语句原理，`contextlib` 简化写法
   - 用于文件、数据库连接、锁等资源管理，避免资源泄漏
5. 模块与包管理
   - 绝对导入 / 相对导入，`__init__.py` 的作用
   - Python 导入机制与命名空间规则

### 三、工程化与生产规范（新手最易忽略，职场必备）

1. 代码规范与类型提示
   - PEP8 编码规范、统一命名风格
   - 类型注解（Type Hints）+ `typing` 模块（`Optional` / `Union` / `List` 等），生产级项目强制要求
2. 环境与依赖管理
   - 虚拟环境：`venv` / `virtualenv` 做环境隔离
   - 依赖管理：`requirements.txt` / `pyproject.toml`，pip 最佳实践
3. 日志系统
   - `logging` 模块：日志级别、处理器、格式化、日志轮转
   - 生产环境禁止用 `print` 输出调试信息
4. 异常处理最佳实践
   - 自定义业务异常、异常分层设计、异常与日志联动
   - 精准捕获、合理抛出，避免吞掉关键异常
5. 单元测试
   - `pytest` 框架，测试用例、断言、fixture、mock
   - 核心业务逻辑必须做单测覆盖

### 四、高频标准库与第三方库

#### 核心标准库（内置免安装，日常高频使用）

- `collections`：`defaultdict` / `Counter` / `deque` / `namedtuple` 等扩展数据结构
- `pathlib`：面向对象的路径处理，替代传统 `os.path`
- `datetime` / `time`：时间处理与时区问题
- `os` / `sys`：系统交互、命令行参数、环境变量
- `json`：序列化与反序列化
- `concurrent.futures`：线程池 / 进程池高级接口
- `re`：正则表达式

#### 通用必备第三方库

- `requests`：HTTP 请求，接口开发 / 爬虫通用
- `pandas` / `numpy`：结构化数据处理与数值计算
- `pydantic`：数据模型与校验，FastAPI 核心依赖
- `SQLAlchemy`：ORM 数据库操作框架

### 五、并发与异步编程（高阶性能优化核心）

1. 并发基础
   - GIL 全局解释器锁的原理与实际影响
   - 选型原则：IO 密集型用多线程，CPU 密集型用多进程
   - 线程安全、锁机制、队列通信
2. 异步编程
   - `asyncio` 核心：`async/await` 语法、事件循环、协程原理
   - `aiohttp` 异步 HTTP 框架，高并发爬虫 / 接口开发
   - 异步编程的常见陷阱与最佳实践

### 六、主流方向深耕（按职业目标选学）

1. Web 后端开发
   - 首选 `FastAPI`（异步、高性能、自动文档，当前工业界主流），其次 Django / Flask
   - RESTful 接口设计、中间件、数据库集成、权限认证
   - 部署：`gunicorn` / `uvicorn` + Nginx + Docker
2. 数据分析 / AI 方向
   - 数据处理：pandas /numpy
   - 可视化：matplotlib /seaborn
   - 机器学习：scikit-learn；深度学习：PyTorch（当前业界主流）
3. 自动化 / 运维方向
   - 脚本自动化：替代 Shell 做批量任务处理
   - 远程操作：`paramiko` / `fabric`
   - 定时任务：`APScheduler`
4. 爬虫方向
   - 基础：requests + BeautifulSoup /parsel
   - 动态页面：Playwright / Selenium
   - 框架：Scrapy，反爬策略与分布式爬虫

### 七、高阶进阶（资深开发核心竞争力）

- 性能优化：代码性能分析（`cProfile`）、内存优化、循环优化、Cython 加速
- 内存管理：引用计数、分代垃圾回收、循环引用、内存泄漏排查
- 设计模式：常用设计模式的 Pythonic 实现
- 部署与运维：Docker 容器化、CI/CD 流程、WSGI/ASGI 协议原理
- 安全：依赖漏洞扫描、注入防护、敏感信息管理

# 基础速通

## 一、核心数据类型（生产高频）

### 1. 基础数值与真值判断

- 类型：`int`整数、`float`浮点数、`bool`布尔值（`True`/`False` 本质是 1/0）
- 生产规范：直接用真值判断，禁止写 `== True`/`== False`
- 假值集合：`0`、`""`空串、`[]`空列表、`{}`空字典、`None`

```
a = 0
if not a:  # 等价 a == 0，通用且简洁
    print("a为假值")
```

### 2. 字符串 string

- 格式化首选 **f-string**（Python3.6+，性能最高、可读性最好）`f"内容,变量用{}包着"`
- 高频方法：`strip`去空格、`split`切割、`join`拼接、`replace`替换
- 切片规则：`[起始:结束:步长]`，左闭右开 步长默认是1 起始和结束默认是首尾

```
name, age = "张三", 25
info = f"姓名：{name}，年龄：{age}"  # 生产级格式化

s = "  hello,world  "
s.strip()       # 去首尾空格 → "hello,world"
s.split(",")    # 按逗号切割 → ["  hello", "world  "]
",".join(["a","b","c"])  # 列表拼字符串 → "a,b,c"

"abcdef"[1:4]   # 切片 → "bcd"
"abcdef"[::-1]  # 反转 → "fedcba"
```

### 3. 列表（有序可变）list

- 生产最佳实践：优先用**列表推导式**生成数据，遍历用 `enumerate` 同时拿索引 + 值
- `range(a,b)` 从a到b-1 左开右闭
- `range(a)` 从0到a>0 且为int

```
lst = [1,2,3]
lst.append(4)  # 末尾追加 返回None
lst.pop()      # 末尾弹出4
print(lst.pop()) # 3

# 生产标准遍历写法
for idx, val in enumerate(lst):
    print(f"索引{idx}: 值{val}")

# 列表推导式（替代简单for循环，性能更高）
squares = [x**2 for x in range(1, 11)]          # 1-10的平方
even_squares = [x**2 for x in range(1, 11) if x % 2 == 0]  # 带条件筛选 1-10中偶数的平方
```

### 4. 元组（有序不可变）tuple

- 核心特性：不可修改，性能优于列表
- 生产用途：**函数多返回值**、**解包赋值**、常量集合

```
# 函数多返回值（本质返回元组）
def get_user():
    return "张三", 25  

name, age = get_user()  # 解包赋值，生产高频写法
```

### 5. 字典（键值对，接口开发核心）dict

- Python3.7+ 天然有序；键必须是不可变类型（字符串 / 数字 / 元组）
- 生产规范：取值用 `.get()`  不用 `字典["键"]` 避免 `KeyError` 崩溃
-  `.get()` 可以当不存在时候有默认值 不写的话返回None
- `字典.items()` 用来遍历字典

```
user = {"name": "张三", "age": 25}

# 安全取值（生产首选）
age = user.get("age", 0)    # 不存在返回默认值0
gender = user.get("gender") # 不存在返回None

# 遍历键值对 字典.items()
for k, v in user.items():
    print(k, v)

# 字典推导式
num_dict = {x: x**2 for x in range(5)}
```

### 6. 集合（无序不重复） set

- 生产核心用途：数据去重、集合运算

```
# 列表去重（注意：会打乱原有顺序）变为从小到大
lst = [1,2,2,3,3,3]
unique_lst = list(set(lst))

a, b = {1,2,3}, {2,3,4}
a & b  # 交集 {2,3}
a | b  # 并集 {1,2,3,4}
a - b  # 差集 {1} 前面有但后面没有的
```

### 变量名的查找优先级规则:LEGB

从内到外、就近匹配 找不到出现 `NameError`

只要在函数内部出现了 `变量 = 值` 的赋值语句，这个变量就会被默认识别为局部变量，覆盖同名全局变量。

四个字母分别对应四层作用域：

| 层级  | 全称      | 含义              | 作用域范围                                          |
| ----- | --------- | ----------------- | --------------------------------------------------- |
| **L** | Local     | 局部作用域        | 函数、`lambda` 表达式内部的变量                     |
| **E** | Enclosing | 嵌套 / 闭包作用域 | 外层嵌套函数的作用域（仅嵌套函数场景存在）          |
| **G** | Global    | 全局作用域        | 单个 `.py` 文件顶层、不在任何函数内的变量           |
| **B** | Built-in  | 内置作用域        | Python 解释器内置的名字（如 `print`、`len`、`int`） |

#### 1. L - Local 局部作用域

- 在**函数体内部**定义的变量（包括函数参数），都属于局部作用域。
- 生命周期：函数被调用时创建，函数执行结束（return / 异常退出）后立即销毁，外部无法访问。

```
def func():
    x = 10  # 局部变量
    print(x)

func()       # 输出 10
print(x)     # 报错：NameError，外部访问不到局部变量
```

#### 2. E - Enclosing 外层嵌套作用域

- 只在**函数嵌套函数**的场景中存在，指的是内层函数的外层函数的作用域。
- 这是闭包、装饰器的核心作用域基础。

```
def outer():
    x = "外层变量"  # 属于 Enclosing 作用域
    def inner():
        # 内层函数查找 x 时，先找自己局部，找不到就去外层找
        print(x)
    inner() #先有内层函数再调用

outer()  # 输出：外层变量
```

#### 3. G - Global 全局作用域

- 在一个 `.py` 文件的**最顶层**定义、不在任何函数 / 类内部的变量，就是全局变量。
- 作用范围：整个当前文件内的所有位置都可以**读取**。

```
x = "全局变量"  # 全局作用域

def func():
    # 函数内部可以直接读取全局变量
    print(x)

func()  # 输出：全局变量
```

#### 4. B - Built-in 内置作用域

- Python 解释器启动就自带的内置名称，所有模块、所有位置都可以直接使用。
- 比如 `print()`、`len()`、`range()`、`int()`、`str()` 等都属于内置作用域。
- 注意：如果自己定义了同名变量（比如 `def print(): pass`），会**遮蔽**内置函数，这是常见坑。

### 修改外层变量：`global` 与 `nonlocal`

如果想在函数内部修改**全局变量**或**外层嵌套变量**，必须显式声明，否则只会创建局部变量。

#### 1. `global`：声明使用全局变量

告诉解释器：这个变量来自全局作用域，赋值时修改的是全局变量，不是新建局部变量。

```
x = 10

def func():
    global x  # 声明 x 是全局变量
    x = 20    # 修改全局变量
    print(x)

func()
print(x)  # 输出 20，全局变量已被修改
```

#### 2. `nonlocal`：声明使用外层嵌套变量

用于嵌套函数中，声明变量来自**最近的外层非全局作用域**，专门用来修改闭包外层的变量。

```
def outer():
    x = 10
    def inner():
        nonlocal x  # 声明 x 来自外层函数
        x = 20      # 修改外层函数的变量
    inner()
    print(x)

outer()  # 输出 20
```

| 关键字     | 作用对象         | 使用场景                 |
| ---------- | ---------------- | ------------------------ |
| `global`   | 全局作用域变量   | 函数内修改全局变量       |
| `nonlocal` | 外层嵌套函数变量 | 内层函数修改外层闭包变量 |

- **代码块不产生新作用域**：`if`/`for`/`while` 内部的变量和外部同属一个作用域，循环结束后变量依然存在。
- **同名变量遮蔽**：内层作用域的同名变量会完全覆盖外层，外层变量不会再被访问到。
- **禁止滥用全局变量**：全局变量生命周期长、多处可修改，容易造成难以调试的副作用，优先用函数参数和返回值传递数据。
- **不要和内置名重名**：比如自定义 `list = [1,2,3]` 会覆盖内置的 `list()` 构造函数，导致后续代码异常。

------

## 二、流程控制

### 1. 条件判断

- 简单逻辑用**三元表达式**简写，复杂逻辑用 `if/elif/else`
- `值1 if 条件 else 值2`

```
# 三元表达式（生产高频简写）
age = 20
status = "成年" if age >= 18 else "未成年"
```

### 2. 循环

- 生产首选 `for` + 迭代对象；`while` 仅用于不确定循环次数的场景
- 关键字：`break` 跳出循环、`continue` 跳过本次

```
for i in range(5):
    if i == 3:
        continue  # 跳过3
    print(i)
```

------

## 三、函数（生产级标准写法）

### 1. 函数定义 + 类型提示

- 生产必备：加**类型提示**，提升可读性、支持 IDE 静态检查、降低联调 bug 率
- 四类参数：位置参数、默认参数、关键字参数、可变参数

```
def calculate_sum(a: int, b: int = 10) -> int:
    """计算两个整数的和（生产规范：加文档字符串说明功能）"""
    return a + b

# 三种调用方式
res1 = calculate_sum(5)         # 默认参数 b=10
res2 = calculate_sum(5, 20)     # 位置传参
res3 = calculate_sum(a=3, b=7)  # 关键字传参，顺序无关
```

### 2. 可变参数

- `*args`：接收任意位置参数，封装为**元组**
- `**kwargs`：接收任意关键字参数，封装为**字典**
- 生产用途：通用工具函数、装饰器封装

```
def print_info(*args, **kwargs):
    print("位置参数:", args)   	# 使用时候不加*
    print("关键字参数:", kwargs)

print_info("张三", 25, city="北京", job="开发")

"""函数结果:
位置参数: ('张三', 25)
关键字参数: {'city': '北京', 'job': '开发'}
"""
```

### 3. 函数一等公民特性

函数和整数、字符串、列表等普通数据类型拥有完全平等的地位，它本质就是一种「值」，具备所有普通数据能做的操作。这是 Python 支持函数式编程、装饰器、闭包等高级特性的底层基础。

### 4. lambda 匿名函数

```
lambda 参数: 返回表达式
```

特点：只能一行、只能一个表达式、不能写 if 块 /for 循环 / 赋值语句

- 适用场景：单行简单逻辑，配合 `sorted`/`map` 等高阶函数
- 生产规范：复杂逻辑必须写命名函数，禁止嵌套 lambda

```
# 按字典的age字段排序（列表数据排序高频写法）
users = [{"name": "张三", "age": 25}, {"name": "李四", "age": 20}]  
users.sort(key=lambda x: x["age"]) #x代表 users 里每一个字典元素
```

------

## 四、文件读写与 JSON（API 开发必备）

### 1. 文件操作：with 上下文管理器

- 生产唯一推荐：`with` 自动关闭文件，避免资源泄漏
- 常用模式：`r`读、`w`覆盖写、`a`追加、`rb`二进制读（图片 / 文件流）

```
# 读取文件
with open("test.txt", "r", encoding="utf-8") as f:
    content = f.read()  # 读取全部；逐行读用 for line in f

# 写入文件
with open("test.txt", "w", encoding="utf-8") as f:
    f.write("hello python")
```

### 2. JSON 序列化 / 反序列化

- 前后端接口数据交互的标准格式，生产 100% 会用到
- `dumps`：Python 对象 → JSON 字符串；
- `loads`：JSON 字符串 → Python 对象

```
import json

# 序列化（接口返回数据用）
user = {"name": "张三", "age": 25}
json_str = json.dumps(user, ensure_ascii=False)  # 保留中文 否则是ascii码

# 反序列化（接收接口参数用）
json_data = '{"name": "李四", "age": 30}'
user_dict = json.loads(json_data) # type是dict
```

#### 标准 JSON 格式规则（必记）

1. 键（key）**必须用双引号 "**，不能单引号；
2. 字符串值也必须双引号；
3. 不能写注释 `//`、不能写函数、不能写 `None`；
4. 支持类型：对象`{}`、数组`[]`、数字、字符串、`true`、`false`、`null`

#### 合法 JSON 示例（纯文本）

```
{
  "name": "张三",
  "age": 20,
  "isStudent": true,
  "hobbies": ["打球", "看书"],
  "address": {
    "city": "北京"
  }
}
```







# 进阶核心（生产级高频用法，覆盖80%企业项目场景）

------

### 1. 异常处理（程序健壮性核心）

##### 核心结构

```
try:
    # 可能出错的代码
    result = 10 / 0
except ZeroDivisionError as e:  # 捕获具体异常，拿到异常对象
    print(f"计算出错: {e}")
except ValueError:
    print("值类型错误")
else:
    # 无异常才执行
    print("运行成功", result)
finally:
    # 无论是否异常都执行（释放资源、关闭连接）
    print("结束")
```

**生产铁律**：

- ❌ 禁止写裸 `except:`，会捕获所有异常包括键盘中断、系统退出
- ❌ 非顶层代码禁止 `except Exception` 全捕获
- ✅ 只捕获你能处理的**具体异常类型**，异常逐层向上抛 
- ✅ 接口/程序入口处统一兜底捕获，返回友好错误

##### 主动抛异常 & 自定义异常

```
# 主动抛出
if age < 0:
    raise ValueError("年龄不能为负数")

# 自定义业务异常（API开发必备，区分错误类型）
class BizException(Exception):
    def __init__(self, code: int, msg: str):
        self.code = code
        self.msg = msg
        super().__init__(msg)

# 使用
raise BizException(4001, "参数校验失败")
```

##### `raise` 用法

 `raise` 是 Python 中**主动抛出异常**的关键字。不同于程序运行出错时自动触发的异常，`raise` 用于在逻辑不满足预期时，手动中断正常流程，抛出指定错误，交由上层调用者通过 `try-except` 处理。

- 抛出异常类（最简写法）: 直接写异常类名，Python 会自动实例化该异常类，默认无自定义错误信息。

```
# 抛出一个值错误异常
raise ValueError
```

​	缺点：没有错误描述，排查问题不直观，实际开发少用。

- 抛出带描述信息的异常实例（最常用）: 给异常类传入字符串参数，附带具体错误原因，捕获后可直接读取错误信息。

```
def divide(a, b):
    if b == 0:
        raise ZeroDivisionError("除数不能为 0")
    return a / b

divide(10, 0)
# 报错：ZeroDivisionError: 除数不能为 0
```

- 抛出已实例化的异常对象: 先创建异常实例，再通过 `raise` 抛出，适合需要给异常附加额外属性的场景。

```
err = TypeError("参数类型必须是整数")
err.error_code = 1001
raise err
```

------

###### 无参 raise：重新抛出当前异常

在 `except` 代码块中，直接写 `raise`（不带任何参数），会**原样重新抛出当前捕获的异常**。

典型场景：先记录日志、做清理工作，再把异常交给上层处理，不吞掉异常。

```
import logging

def save_data(data):
    try:
        # 模拟写入文件
        f = open("data.txt", "r") 	# 模式不对会报错
        f.write(data)
    except Exception as e:
        # 先记录错误日志
        logging.error(f"保存数据失败: {e}")
        # 原样抛出异常，让上层知道出错了
        raise

save_data("123")  # 日志记录后，异常依然会向上抛出
```

> 注意：无参 `raise` 只能在 `except` 块内使用，在普通代码中使用会触发 `RuntimeError: No active exception to reraise`。

------

###### 四、自定义异常 + raise

内置异常类型有限，实际项目中通常会自定义业务异常，用来区分「系统错误」和「业务规则错误」，便于分层捕获和处理。

#### 1. 定义自定义异常

必须继承 `Exception`（不要继承 `BaseException`，避免捕获到系统退出、键盘中断等非业务异常）。

```
# 自定义业务异常基类
class BusinessError(Exception):
    """业务逻辑异常基类"""
    pass

# 具体业务异常
class InsufficientBalanceError(BusinessError):
    """余额不足异常"""
    pass

class UserNotFoundError(BusinessError):
    """用户不存在异常"""
    pass
```

#### 2. 抛出与捕获自定义异常

```
def pay(user_id, amount):
    balance = get_user_balance(user_id)
    if balance is None:
        raise UserNotFoundError(f"用户 {user_id} 不存在")
    if balance < amount:
        raise InsufficientBalanceError(f"余额不足，当前余额: {balance}")
    # 执行扣款逻辑...

# 上层统一捕获业务异常
try:
    pay(1001, 500)
except BusinessError as e:
    print(f"业务处理失败: {e}")
```

------

###### 五、异常链：raise ... from ...

Python 3 引入的特性，用于关联异常的因果关系：**抛出新异常时，保留原始异常的上下文**。调试时会清晰显示「A 异常导致了 B 异常」，便于定位根因。

#### 1. 基础用法：关联原始异常

```
try:
    int("abc")  # 原始异常：ValueError
except ValueError as e:
    # 抛出包装后的业务异常，同时关联原始异常
    raise RuntimeError("用户ID格式转换失败") from e
```

报错堆栈会显示：

```
The above exception was the direct cause of the following exception:
RuntimeError: 用户ID格式转换失败
```

#### 2. `from None`：抑制异常链

如果不想暴露底层异常细节（比如隐藏内部实现），可以用 `from None` 切断异常链，只显示外层异常。

```
try:
    1 / 0
except ZeroDivisionError:
    # 只抛出包装后的异常，隐藏底层的 ZeroDivisionError
    raise ValueError("计算参数非法") from None
```

------

###### 最佳实践与常见误区

1. 核心使用场景

- **入口参数校验**：函数开头校验参数合法性，不合法直接抛出，避免后续逻辑执行到一半出错
- **业务规则不满足**：如库存不足、权限不够、状态非法等业务场景
- **异常包装**：底层技术异常包装为业务异常后抛出，解耦底层实现与上层逻辑

2. 最佳实践

- **精准抛出异常类型**：优先用内置的具体异常（`ValueError`、`TypeError`、`KeyError` 等），不要一上来就 `raise Exception`
- **错误信息带上下文**：不要只写「参数错误」，要说明哪个参数、是什么值、为什么错
- **底层抛出，上层处理**：底层函数只负责抛出异常，业务上层统一捕获处理，不要在底层吞掉异常
- **自定义异常继承 Exception**：不要继承 `BaseException`，避免误捕获系统级异常

3. 常见误区

- `raise` 是关键字不是函数，不需要加括号：`raise ValueError("msg")` 是实例化异常类，不是调用 raise 函数
- `raise` 之后的代码永远不会执行，和 `return` 一样，属于流程终止语句
- 不能抛出普通对象，只能抛出 `BaseException` 子类的实例，否则触发 `TypeError`

------

`raise` 就是手动报错的工具：简单场景抛内置异常加描述，业务场景用自定义异常，需要透传异常用无参 raise，需要关联根因用异常链。



## Python 内置异常分类

#### 一、基类（所有异常的父类）

1. `BaseException`：顶层基类，包含系统退出类异常
2. `Exception`：常规程序异常父类（**日常捕获都继承它**）

#### 二、运行时常见业务异常（重点）

##### 1. 语法 / 逻辑类

- `SyntaxError`：代码语法错误（无法被 try 捕获，执行前就报错）
- `IndentationError`：缩进错误
- `NameError`：使用未定义变量
- `UnboundLocalError`：局部变量引用前未赋值

##### 2. 类型与数值错误

- `TypeError`：类型不匹配（数字 + 字符串相加）
- `ValueError`：值非法（int ("abc")）
- `IndexError`：列表 / 元组下标越界
- `KeyError`：字典键不存在
- `AttributeError`：对象没有该属性 / 方法
- `ZeroDivisionError`：除零错误
- `OverflowError`：数值溢出

##### 3. 容器与迭代

- `StopIteration`：迭代器耗尽
- `AssertionError`：assert 断言失败

##### 4. 文件、IO、操作系统

- `FileNotFoundError`：文件不存在
- `PermissionError`：权限不足
- `IsADirectoryError`：把文件夹当文件打开
- `OSError`：操作系统底层错误（以上三个都是它子类）
- `IOError`：IO 读写异常（Python3 并入 OSError）

##### 5. 导入与模块

- `ImportError`：导入模块失败
- `ModuleNotFoundError`：模块找不到（子类）

##### 6. 多线程 / 内存 / 运行

- `RecursionError`：递归深度超限
- `MemoryError`：内存耗尽
- `TimeoutError`：超时错误

#### 三、系统退出类（不建议捕获）

继承自 `BaseException`，不属于 `Exception`：

- `KeyboardInterrupt`：Ctrl+C 手动终止程序
- `SystemExit`：sys.exit () 退出
- `GeneratorExit`：生成器关闭

#### 四、自定义异常

继承 `Exception` 实现：

```
class MyError(Exception):
    pass
```

#### 最简捕获层级写法

```
try:
    pass
except ValueError:
    pass
except Exception as e:  # 兜底所有业务异常
    pass
```

------

### 2. 面向对象编程（工程化代码基石）

##### 基础类 & 封装

```
class User:
    # 类属性（所有实例共享）
    type = "普通用户"

    def __init__(self, name: str, age: int):
        # 实例属性（每个实例独有）
        self.name = name
        self._age = age  # _开头约定为"私有"，仅规范，非强制

    def say_hello(self) -> str:
        return f"我是{self.name}"

# 实例化
u = User("张三", 25)
u.say_hello()
```

##### 构造函数 `__init__`

**作用**：对象**创建完成后初始化**，给实例属性赋值，不是真正创建对象。触发时机：`obj = Class()` 实例化时自动调用。

```
class User:
    # 构造方法
    def __init__(self, name, age):
        self.name = name   # 实例属性
        self.age = age

u = User("张三", 18)
print(u.name)  # 张三
```

关键点：

- 第一个参数必须是 `self`，代表当前实例对象；
- 不能有返回值（return）；
- 仅负责初始化，真正分配内存由 `__new__` 完成（极少重写）。

##### 生产神器：数据类 `dataclasses`

替代手写繁琐的 `__init__`，定义数据模型/接口实体首选

```
from dataclasses import dataclass

@dataclass
class UserDTO:
    name: str
    age: int
    city: str = "北京"  # 默认值

# 自动生成 __init__、__repr__、__eq__ 等方法
u = UserDTO("李四", 30)
print(u.name)
```

##### 继承 & 多态

```
class Admin(User):
    def say_hello(self) -> str:
        return f"管理员{self.name}"

# super() 调用父类方法
class VipUser(User):
    def __init__(self, name: str, age: int, level: int):
        super().__init__(name, age)
        self.level = level
```

##### 常用魔术方法

- `__str__`：打印对象时的可读描述 让对象实例可以直接print `ruturn <要输出的字符串>`

- `__repr__`：调试用的官方描述 要返回直接能直接 eval 重建对象的字符串

  `eval(字符串)`：把一段符合 Python 语法的字符串，当成代码直接执行，并返回执行结果。很不安全

- `__len__`：支持 `len()` 调用

- `__eq__` : 等于判断 `==`  重写后自定义两个对象相等规则，默认是比较内存地址。 `is` 永远判断内存地址，不受影响。

------

### 3. 进阶核心特性

#### ① 生成器（处理大数据流必备）

用 `yield` 替代 `return`，按需生成数据，**不占内存**，生产中处理大文件、海量数据必用

函数里只要出现 `yield`，调用返回生成器对象，**不会执行函数内部代码**

```
def read_big_file(file_path: str):
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            yield line.strip()  # 逐行返回，不一次性加载全部

# 使用
for line in read_big_file("data.txt"):
    process(line)

# 生成器表达式（类似列表推导，省内存）
gen = (x**2 for x in range(1000000))
```

**四种遍历方式:**

1. `next (<生成器实例>)` 逐个取，耗尽抛 StopIteration

2. for 循环（最常用，自动捕获结束异常）`for item in <生成器函数或者实例>`

3. 转列表 / 元组一次性取出（小数据可用）

4. itertools.islice 截取前 N 个（大数据不全部加载）

   ```
   from itertools import islice
   g = (i for i in range(100000))
   # 只取前5个，后面完全不计算
   list(islice(g, 5))
   ```

##### 1. 本质再明确

- 生成器 = **惰性迭代器**，天然符合迭代器协议，可直接被 `for` 遍历
- 核心优势：**按需计算、极低内存**，数据量越大，内存优势越明显
- 执行规则：调用生成器函数本身不执行任何代码，只有迭代时（`for`/`next()`/`list()`）才逐次触发执行

##### 2. 生产级三大高频场景

场景 1：大文件 / 海量数据逐行处理（后端 / 运维必备）

适用：GB 级日志、CSV 批量导入、数据库全表导出，内存稳定在 KB 级，不会 OOM(out of memory)

```
def iter_large_file(file_path: str):
    """按行读取大文件，对上层屏蔽内存细节"""
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            yield line.strip()

# 业务代码：统计错误日志，10G文件也不会爆内存
error_count = 0
for line in iter_large_file("app.log"):
    if "ERROR" in line:
        error_count += 1
```

场景 2：分页接口无缝封装（对接第三方 API 标准写法）

适用：对接外部分页接口，对上层屏蔽分页逻辑，业务代码像遍历列表一样使用

```
import requests

def iter_user_list(page_size: int = 100):
    """分页拉取全量用户，封装为生成器，上层无需关心分页"""
    page = 1
    while True:
        resp = requests.get(
            "https://api.example.com/users",
            params={"page": page, "page_size": page_size},
            timeout=10
        )
        data = resp.json()["data"]
        if not data:
            break
        yield from data  # 生产标准简写：等价于 for item in data: yield item
        page += 1

# 业务代码：完全感知不到分页
for user in iter_user_list():
    sync_user_to_local(user)
```

> `yield from` 是嵌套生成器 / 可迭代对象的语法糖，代码更简洁，异常传递也更完整，生产首选。

场景 3：生成器管道（数据流链式处理，ETL / 数据清洗常用）

思想：每个步骤一个生成器，像流水线一样串联，全程惰性执行，不产生任何中间列表，内存零增长

```
def read_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        yield from f

def filter_error(lines):
    for line in lines:
        if "ERROR" in line:
            yield line

def parse_time(lines):
    for line in lines:
        time_str = line.split(" ")[0]
        yield time_str, line

# 管道串联：逐行流动，全程无全量数据加载
lines = read_lines("app.log")
error_lines = filter_error(lines)
result = parse_time(error_lines)

for t, line in result:
    save_to_db(t, line)
```

##### 3. 生产避坑与最佳实践

✅ 必守规则

1. 生成器**只能遍历一次**，遍历结束即为耗尽状态；需重复使用请重新调用函数创建新对象。
2. 仅在**数据量大、无需随机访问**时使用；小数据量直接用列表，可读性优先。
3. 不在生成器内做复杂副作用操作（修改全局变量、写库），惰性执行会导致执行时机不可控。

❌ 常见坑

- 用 `list(生成器)` 强转：直接失去内存优势，等于白写，除非确实需要最终结果列表。
- 把生成器赋值给变量后多次遍历：**第二次遍历直接为空，排查难度高。**

#### ② 装饰器（AOP切面编程，日志/计时/权限）

**标准无参装饰器写法**（必须加 `functools.wraps` 保留原函数元信息）

```
import functools
import time

def timer(func):
    @functools.wraps(func)  # 生产规范：必须加，否则函数名、文档会丢失
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print(f"{func.__name__} 耗时: {time.time()-start:.2f}s")
        return result
    return wrapper

# 使用
@timer
def query_data():
    time.sleep(1)
```

生产常见用途：接口鉴权、日志埋点、参数校验、缓存。

#### ③ 上下文管理器（资源自动释放）

`with` 语句的底层实现，除了文件操作，自定义数据库连接、HTTP会话都用它

```
# 写法1：类实现
class DBConnection:
    def __enter__(self):
        print("建立连接")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        print("关闭连接")
        # 返回True会吞掉异常，一般不建议

# 写法2：装饰器简写（生产更常用）
from contextlib import contextmanager

@contextmanager
def db_connect():
    print("建立连接")  # __enter__
    try:
        yield "连接对象"  # 产出资源给with后的变量
    finally:
        print("关闭连接")  # __exit__

# 使用
with db_connect() as conn:
    print("执行SQL")
```

------

### 4. 高频标准库 & 必备第三方库

##### ① 路径处理：`pathlib`（现代写法，替代 `os.path`）

```
from pathlib import Path

# 当前目录
base = Path.cwd()
# 路径拼接（自动适配Windows/Mac）
file_path = base / "data" / "user.json"

file_path.exists()    # 是否存在
file_path.is_file()   # 是否是文件
file_path.parent      # 父目录
file_path.suffix      # 后缀名 .json
```

##### ② 时间处理：`datetime`

```
from datetime import datetime, timedelta

# 当前时间
now = datetime.now()
# 时间转字符串
now.strftime("%Y-%m-%d %H:%M:%S")
# 字符串转时间
datetime.strptime("2024-01-01", "%Y-%m-%d")
# 时间加减
tomorrow = now + timedelta(days=1)
# 时间戳互转
timestamp = now.timestamp()
datetime.fromtimestamp(timestamp)
```

##### ③ 正则：`re`

```
import re

# 匹配手机号
pattern = r"^1[3-9]\d{9}$"
re.match(pattern, "13800138000")

# 提取所有数字
re.findall(r"\d+", "价格199元，优惠20元")  # ['199', '20']

# 替换
re.sub(r"\s+", "", "a b  c   d")  # 去所有空格 → "abcd"
```

##### ④ HTTP请求：`requests`（第三方库，调用外部接口必备）

```
import requests

# GET请求
resp = requests.get("https://api.example.com/user", params={"id": 1})
resp.status_code    # 状态码
resp.json()         # 解析JSON响应体

# POST请求（JSON格式）
resp = requests.post(
    "https://api.example.com/user",
    json={"name": "张三", "age": 25},
    timeout=10  # 超时时间，生产必加
)
```

------





# 补充

Python 中非常经典的**带索引遍历**写法。拆解开来，它做了三件事：

**1. 遍历 prompts 列表**
`for ... in prompts` 表示依次取出 `prompts` 列表中的每一个元素（即每一个提示词）。

**2. 使用 enumerate 获取“序号”**
`enumerate(prompts, 1)` 是一个内置函数，它会为列表中的每个元素自动生成一个**计数器**（索引）。

返回索引在前 元素在后

**3. start=1 让序号从 1 开始**
括号里的 `1` 是 `enumerate` 的起始参数，表示计数器从 **1** 开始，而不是默认的 0。

`for i, prompt in enumerate(prompts, 1)`