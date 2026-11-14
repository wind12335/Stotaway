# GitHub Deploy Key 与 NVIDIA 分支推送

本文件记录仓库 `wind12335/Stotaway` 的 NVIDIA 分支推送方式。它只包含**公钥**和操作说明；私钥不进入 Git 仓库。

## 当前信息

| 项目 | 内容 |
|---|---|
| GitHub 仓库 | https://github.com/wind12335/Stotaway |
| 目标分支 | `nvidia` |
| 已准备的提交 | `d55af53ab841cdb95349198084c6f0082fe3c597`，包含 NVIDIA 代码、实验及主要原始数据 |
| 公钥指纹 | `SHA256:bcnuObJSpsoaBlYQui7vcfs1hP3AAMHqCUmUX2I9WLw`（ED25519） |
| 本机公钥 | `/root/.ssh/stotaway_nvidia_deploy.pub` |
| 本机私钥 | `/root/.ssh/stotaway_nvidia_deploy` |
| 上次验证状态 | GitHub SSH 可连接，但公钥尚未获仓库授权，返回 `Permission denied (publickey)`。 |

本机私钥保存在 `/root/.ssh`，不要复制到仓库、文档、聊天或提交中。换机器或重建环境后，本地密钥文件可能不再存在；届时使用已安全保存的原私钥，或重新生成新密钥并在 GitHub 更新授权。

## 1. 添加公钥并授予写权限

打开仓库的 [Deploy Keys 设置](https://github.com/wind12335/Stotaway/settings/keys)，选择 **Add deploy key**，标题可填 `stotaway-nvidia-deploy-20261114`，粘贴以下公钥，**勾选 Allow write access**，然后保存：

```text
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIACrV2loF6kaf5wcytGm4ZP/d+X1Iru15tJmdp8VZk2v stotaway-nvidia-deploy-20261114
```

公钥可以放进仓库；写权限在 GitHub 仓库设置中赋予。Deploy Key 对该仓库生效，并不自动获得其他仓库权限。若账户无权修改仓库 Deploy Keys，需要由仓库管理员操作。

## 2. 检查本地密钥和远端分支

```bash
cd /root/Stotaway
ssh-keygen -lf /root/.ssh/stotaway_nvidia_deploy.pub
git status --short --branch
git log -1 --oneline
```

显式指定本密钥、通过 SSH 查询远端：

```bash
export GIT_SSH_COMMAND='ssh -i /root/.ssh/stotaway_nvidia_deploy -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=8'
git ls-remote --heads git@github.com:wind12335/Stotaway.git
```

如果仍显示 `Permission denied (publickey)`，先核对粘贴的公钥及写权限。如果连接失败，排查网络；本机之前的 HTTPS 代理访问被重置，直接 SSH 曾能连接到 GitHub。不要将令牌写入远端 URL。

## 3. 推送并验证

先查看第 2 步的输出。若远端尚无 `nvidia` 分支，可直接推送；若已存在，先 fetch 并确认远端提交是本地分支的祖先，再进行正常推送。不要用 force 覆盖远端已有工作。

```bash
cd /root/Stotaway
export GIT_SSH_COMMAND='ssh -i /root/.ssh/stotaway_nvidia_deploy -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=8'
git fetch git@github.com:wind12335/Stotaway.git refs/heads/nvidia:refs/remotes/origin/nvidia
# 如果 fetch 报远端分支不存在，可在确认 ls-remote 无该分支后跳过本行。
git merge-base --is-ancestor refs/remotes/origin/nvidia HEAD
# 仅在远端分支存在且上一行退出码为 0 时继续；远端分支不存在时该检查不适用。
git push git@github.com:wind12335/Stotaway.git HEAD:refs/heads/nvidia
git ls-remote git@github.com:wind12335/Stotaway.git refs/heads/nvidia
git rev-parse HEAD
```

核对最后两条命令输出的提交 SHA 相同。此方式不修改现有 HTTPS `origin`；若需要自动设置 upstream，可在确认 SSH 已可用且了解仓库配置后另行设置。
