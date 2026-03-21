/**
 * 文本分割策略
 */
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { getEncodingNameForModel, getEncoding } from "js-tiktoken";
import { Document } from "@langchain/core/documents";
import { logSplitDocumentsInfo } from "./RecursiveCharacterTextSplitter";

/**
 * 获取不同AI 大模型的编码名称
 * 并且基于编码名称获取编码对象
 * 最后基于编码对象计算文本的Token数
 * 同样是两个汉字苹果和吃饭，返回的token数可能不同，因此字符个数和token个数并不是一一对应的关系
 * 这和不同模型的分词器有关
 */
const modelName = "gpt-4";
const encodingName = getEncodingNameForModel(modelName); // 'cl100k_base'
const encoding = getEncoding(encodingName); // return Encoding object
const jsCode = `
    // Complete shopping cart implementation
    class Product {
      constructor(id, name, price, description) {
        this.id = id;
        this.name = name;
        this.price = price;
        this.description = description;
      }

      getFormattedPrice() {
        return '$' + this.price.toFixed(2);
      }
    }

    class ShoppingCart {
      constructor() {
        this.items = [];
        this.discountCode = null;
        this.taxRate = 0.08;
      }

      addItem(product, quantity = 1) {
        const existingItem = this.items.find(item => item.product.id === product.id);
        if (existingItem) {
          existingItem.quantity += quantity;
        } else {
          this.items.push({ product, quantity, addedAt: new Date() });
        }
        return this;
      }

      removeItem(productId) {
        this.items = this.items.filter(item => item.product.id !== productId);
        return this;
      }

      calculateSubtotal() {
        return this.items.reduce((total, item) => {
          return total + (item.product.price * item.quantity);
        }, 0);
      }

      calculateTotal() {
        const subtotal = this.calculateSubtotal();
        const discount = this.calculateDiscount();
        const tax = (subtotal - discount) * this.taxRate;
        return subtotal - discount + tax;
      }

      calculateDiscount() {
        if (!this.discountCode) return 0;
        const discounts = { 'SAVE10': 0.10, 'SAVE20': 0.20, 'WELCOME': 0.15 };
        return this.calculateSubtotal() * (discounts[this.discountCode] || 0);
      }
    }

    // Usage example
    const product1 = new Product(1, 'Laptop', 999.99, 'High-performance laptop');
    const product2 = new Product(2, 'Mouse', 29.99, 'Wireless mouse');
    const cart = new ShoppingCart();
    cart.addItem(product1, 1).addItem(product2, 2);
    console.log('Total:', cart.calculateTotal());
`;
const jsCodeDocuments = new Document({
  pageContent: jsCode,
  metadata: {
    source: "日志文件",
    date: "2024-01-15",
  },
});

/**
 * RecursiveCharacterTextSplitter优先使用\n 换行符分割
 * 当分割后的chunk大小还是大于chunkSize时，再使用其他分隔符递归分割
 * chunksize默认指的是字符数，而不是Token数
 * 但是可以通过指定lengthFunction函数来设置成为基于Token数来分割
 */
const jsCodeSplitter = RecursiveCharacterTextSplitter.fromLanguage("js", {
  chunkSize: 200, // 每个chunk的字符数
  chunkOverlap: 20, // 每个chunk之间的重叠字符数
});

const splitDocuments = await jsCodeSplitter.splitDocuments([jsCodeDocuments]);
console.log(`JavaScript代码文档分割完成，共 ${splitDocuments.length} 个分块\n`);

logSplitDocumentsInfo(splitDocuments);
