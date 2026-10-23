# 教程视频源码

需要 Node.js 22+。在此目录执行 `npm ci`，之后 `npm run studio` 预览，`npm run render` 导出到 `../media/mathknow-tutorial.mp4`。首次渲染需要下载 Chrome Headless Shell，或使用 Remotion 的 `--browser-executable` 指定本机兼容浏览器。

视频时长 120 秒，1600×900，24 fps，中文无声动画；每章节 10 秒。修改 `src/index.tsx` 中的 slides 和界面示意即可更新内容；修改总章节数时同步更新时长和页面中的章节数。

`public/NotoSansSC.ttf` 为 Noto Sans SC 的字形子集，使用 SIL OFL 1.1；许可证见 `public/OFL.txt`。若增加当前子集不含的新汉字，应从原字体重新生成子集。字体来源：https://github.com/google/fonts/tree/main/ofl/notosanssc 。不包含密钥、用户会话或浏览器资料。
