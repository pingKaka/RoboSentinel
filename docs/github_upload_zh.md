# 将 RoboSentinel 上传到 GitHub

GitHub只保存代码。模型权重、采集视频、运行结果和密钥不要上传。

## 1. 创建GitHub仓库

在GitHub点击 **New repository**，仓库名填写 `RoboSentinel`。不要勾选初始化README，然后创建仓库。

## 2. 检查上传内容

在项目根目录执行：

```bash
cd /home/kakaping/ur_data_collection/RoboSentinel
git status
git check-ignore checkpoint video result/input_frames .env
```

第二条命令应列出存在的权重、视频、结果或密钥目录。提交前务必用 `git status` 确认没有大文件和隐私数据。

## 3. 配置GitHub连接

当前项目的 `origin` 指向原始项目，先将它保留为 `upstream`，再添加自己的仓库：

```bash
git remote rename origin upstream
git remote add origin git@github.com:YOUR_NAME/RoboSentinel.git
git remote -v
```

将 `YOUR_NAME` 替换为你的GitHub用户名。

若未配置SSH，可使用HTTPS：

```bash
git remote add origin https://github.com/YOUR_NAME/RoboSentinel.git
```

HTTPS推送时密码位置应填写GitHub Personal Access Token，不是账号密码。

## 4. 提交并上传

```bash
git add .
git status
git commit -m "Refactor RoboSentinel streaming pipeline"
git branch --show-current
git push -u origin HEAD
```

`git push -u origin HEAD` 会上传当前分支，不要求分支名必须是 `main`。

## 5. 后续更新

每次修改后执行：

```bash
git add .
git status
git commit -m "说明本次修改"
git push
```

模型权重放在 `checkpoint/`，采集视频放在 `video/`。它们通过 `rsync` 单独传到服务器，不经过GitHub。
