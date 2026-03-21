/**
 * 每日 NBA 新闻摘要
 * 1. 首先从WEB网页获取最新的NBA新闻
 */

import { chromium } from "playwright";
import fs from "fs";

// Playwright 最小 demo：加载 NBA 中国官网新闻页面
async function loadNBANews() {
  // 启动浏览器
  const browser = await chromium.launch({ headless: false }); // 暂时使用非无头模式以便观察
  const page = await browser.newPage();

  try {
    // 导航到 NBA 中国官网新闻页面
    await page.goto("https://china.nba.cn/team/1610612747/news", {
      waitUntil: "domcontentloaded", // 等待 DOM 加载完成
      timeout: 60000, // 超时设置
    });

    // 等待几秒钟让页面完全加载
    await page.waitForTimeout(5000);

    // 截图查看页面结构
    await page.screenshot({ path: "./nba-page-screenshot.png" });
    console.log("已保存页面截图到 nba-page-screenshot.png");

    // 获取页面 HTML 查看结构
    const html = await page.content();
    fs.writeFileSync("./nba-page.html", html);
    console.log("已保存页面 HTML 到 nba-page.html");

    // 尝试提取新闻数据（使用更通用的选择器）
    const newsList = await page.evaluate(() => {
      // 尝试不同的选择器
      const selectors = [".news-item", ".article-item", ".item", "li"];
      let items = [];

      for (const selector of selectors) {
        items = Array.from(document.querySelectorAll(selector));
        if (items.length > 0) {
          console.log(`Found ${items.length} items with selector: ${selector}`);
          break;
        }
      }

      return items
        .map((item) => {
          // 尝试不同的标题选择器
          const titleSelectors = [".title", ".news-title", "h3", "h2", "a"];
          let title = "";
          for (const ts of titleSelectors) {
            const el = item.querySelector(ts);
            if (el) {
              title = el.textContent?.trim() || "";
              if (title) break;
            }
          }

          // 尝试获取链接
          const link = item.querySelector("a")?.href || "";

          // 尝试获取日期
          const dateSelectors = [".date", ".news-date", ".time"];
          let date = "";
          for (const ds of dateSelectors) {
            const el = item.querySelector(ds);
            if (el) {
              date = el.textContent?.trim() || "";
              if (date) break;
            }
          }

          return { title, link, date };
        })
        .filter((item) => item.title && item.link);
    });

    // 打印提取的新闻
    console.log("NBA 新闻列表：");
    newsList.forEach((news, index) => {
      console.log(`${index + 1}. ${news.title}`);
      console.log(`   链接: ${news.link}`);
      console.log(`   日期: ${news.date}`);
      console.log("---");
    });

    // 保存到本地文件
    fs.writeFileSync("./nba-news.json", JSON.stringify(newsList, null, 2));
    console.log("新闻已保存到 nba-news.json");
  } catch (error) {
    console.error("加载新闻失败:", error);
  } finally {
    // 关闭浏览器
    await browser.close();
  }
}

// 执行函数
loadNBANews();
