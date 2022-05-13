#! /usr/bin/env python
# -*- coding: utf-8 -*-

class PET_Prompt():

    def __init__(self, dataset_name=""):
        self.label_texts, self.template = [], ""
        if (dataset_name in ['SST-2', 'MR', 'CR']):
            self.label_texts = ["terrible", "great"]
            self.template = "[sentence1] It was [label]."

        elif (dataset_name == 'Subj'):
            self.label_texts = ["subjective", "objective"]
            self.template = "[sentence1] This is [label]."

        elif (dataset_name == 'MPQA'):
            self.label_texts = ["terrible", "great"]
            self.template = "[sentence1] It was [label]."

        elif (dataset_name == 'SST-5'):
            self.label_texts = ["terrible", "bad", "okay", "good", "great"]
            self.template = "[sentence1] It was [label] ."

        elif (dataset_name == 'AGNews'):
            self.label_texts = ["political", "sports", "business", "technology"]
            self.template = "A [label] news : [sentence1] "

        elif (dataset_name == 'Yahoo'):
            self.label_texts = ["Society", "Science", "Health", "Education", "Computer", "Sports", "Business",
                                "Entertainment", "Relationship", "Politics"]
            self.template = "[label] question: [sentence1]"

        elif (dataset_name in ['EPRSTMT']):
            self.label_texts = ["差", "好"]
            self.template = "这次买的东西很[label]. [sentence1]"

        elif (dataset_name in ['TNEWS', 'TNEWSK']):
            self.label_texts = ["故事", "文化", "娱乐", "体育", "财经", "房产", "汽车", "教育", "科技", "军事", "旅游",
                                "国际", "股票", "农业", "电竞"]
            self.template = "这是一则[label]新闻. [sentence1]"

        elif (dataset_name in ['CSLDCP']):
            self.label_texts = ["材料", "作物", "口腔", "药学", "教育", "水利", "理经", "食品", "兽医", "体育", "核能", "力学", "园艺", "水产",
                                "法学", "地质", "能源", "农林", "通信", "情报", "政治", "电气", "海洋", "民族", "航空", "化工", "哲学", "卫生",
                                "艺术", "农工", "船舶", "计科", "冶金", "交通", "动力", "纺织", "建筑", "环境", "公管", "数学", "物理", "林业",
                                "心理", "历史", "工商", "应经", "中医", "天文", "机械", "土木", "光学", "地理", "农资", "生物", "兵器", "矿业",
                                "大气", "医学", "电子", "测绘", "控制", "军事", "语言", "新闻", "社会", "地球", "植物",
                                ]

            self.template = "这是一篇[label]类论文. [sentence1]"

        elif (dataset_name in ['IFLYTEK']):
            self.label_texts = ["打车", "地图", "免费", "租车", "同城", "快递", "婚庆", "家政", "交通", "政务", "社区", "赚钱", "魔幻", "仙侠",
                                "卡牌", "飞行", "射击", "休闲", "动作", "体育", "棋牌", "养成", "策略", "竞技", "辅助", "约会", "通讯", "工作",
                                "论坛", "婚恋", "情侣", "社交", "生活", "博客", "新闻", "漫画", "小说", "技术", "教辅", "问答", "搞笑", "杂志",
                                "百科", "影视", "求职", "兼职", "视频", "短视", "音乐", "直播", "电台", "唱歌", "两性", "小学", "职考", "公务",
                                "英语", "在线", "教育", "成人", "艺术", "语言", "旅游", "预定", "民航", "铁路", "酒店", "行程", "民宿", "出国",
                                "工具", "亲子", "母婴", "驾校", "违章", "汽车", "买车", "养车", "行车", "租房", "买房", "装修", "电子", "挂号",
                                "养生", "医疗", "减肥", "美妆", "菜谱", "餐饮", "资讯", "运动", "支付", "保险", "股票", "借贷", "理财", "彩票",
                                "记账", "银行", "美颜", "影像", "摄影", "相机", "绘画", "二手", "电商", "团购", "外卖", "票务", "超市", "购物",
                                "笔记", "办公", "日程", "女性", "经营", "收款", "其他",
                                ]
            self.template = "这是一款[label]类软件. [sentence1]"

        try:
            self.is_pre =True if self.template.index("[label]") <= self.template.index("[sentence1]") else False
        except:
            print("Prompt template error!")

class NSP_Prompt():

    def __init__(self, dataset_name=""):
        self.label_texts, self.template = [], ""
        self.label_num = 0
        if (dataset_name in ['SST-2', 'MR']):
            self.label_texts = ["terrible", "great"]
            self.template = "A [label] piece of work"
            self.is_pre = True

        elif (dataset_name in ['CR']):
            self.label_texts = ["terrible", "great"]
            self.template = "It was [label]"
            self.template = "A [label] piece of work"
            self.is_pre = True

        elif (dataset_name == 'Subj'):
            self.label_texts = ["subjective", "objective"]
            self.template = "A [label] comment"
            self.is_pre = True

        elif (dataset_name == 'MPQA'):
            self.label_texts = ["negative", "positive"]
            self.template = "It is [label]"
            self.is_pre = True

        elif (dataset_name == 'AGNews'):
            self.label_texts = ["political", "sports", "business", "technology"]
            self.template = "A [label] news :" # template1
            self.is_pre = True

        elif (dataset_name == 'Yahoo'):
            self.label_texts = ["Society", "Science", "Health", "Education", "Computer", "Sports", "Business",
                                "Entertainment", "Relationship", "Politics"]
            self.template = "[label] question :"
            self.is_pre = True

        elif (dataset_name in ['EPRSTMT']):
            self.label_texts = ["很差", "很好"]
            self.template = "这次买的东西[label]"
            self.is_pre = True

        elif (dataset_name in ['TNEWS', 'TNEWSK']):
            self.label_texts = ["故事", "文化", "娱乐", "体育", "财经", "房产", "汽车", "教育", "科技", "军事", "旅游",
                                "国际", "股票", "农业", "电竞"]
            self.template = "这是一则[label]新闻"
            self.is_pre = True

        elif (dataset_name in ['CSLDCP']):
            self.label_texts = ["材料科学与工程", "作物学", "口腔医学", "药学", "教育学", "水利工程", "理论经济学", "食品科学与工程", "畜牧学/兽医学", "体育学",
                                "核科学与技术", "力学", "园艺学", "水产", "法学", "地质学/地质资源与地质工程", "石油与天然气工程", "农林经济管理", "信息与通信工程",
                                "图书馆、情报与档案管理", "政治学", "电气工程", "海洋科学", "民族学", "航空宇航科学与技术", "化学/化学工程与技术", "哲学",
                                "公共卫生与预防医学", "艺术学", "农业工程", "船舶与海洋工程", "计算机科学与技术", "冶金工程", "交通运输工程", "动力工程及工程热物理",
                                "纺织科学与工程", "建筑学", "环境科学与工程", "公共管理", "数学", "物理学", "林学/林业工程", "心理学", "历史学", "工商管理",
                                "应用经济学", "中医学/中药学", "天文学", "机械工程", "土木工程", "光学工程", "地理学", "农业资源利用", "生物学/生物科学与工程",
                                "兵器科学与技术", "矿业工程", "大气科学", "基础医学/临床医学", "电子科学与技术", "测绘科学与技术", "控制科学与工程", "军事学",
                                "中国语言文学", "新闻传播学", "社会学", "地球物理学", "植物保护"]
            self.template = "这是一篇[label]类论文"
            self.is_pre = True

        elif (dataset_name in ['IFLYTEK']):
            self.label_texts = ["打车", "地图导航", "免费WIFI", "租车", "同城服务", "快递物流", "婚庆", "家政", "公共交通", "政务", "社区服务", "薅羊毛",
                                "魔幻", "仙侠", "卡牌", "飞行空战", "射击游戏", "休闲益智", "动作类", "体育竞技", "棋牌中心", "经营养成", "策略", "MOBA",
                                "辅助工具", "约会社交", "即时通讯", "工作社交", "论坛圈子", "婚恋社交", "情侣社交", "社交工具", "生活社交", "微博博客", "新闻",
                                "漫画", "小说", "技术", "教辅", "问答交流", "搞笑", "杂志", "百科", "影视娱乐", "求职", "兼职", "视频", "短视频", "音乐",
                                "直播", "电台", "K歌", "成人", "中小学", "职考", "公务员", "英语", "视频教育", "高等教育", "成人教育", "艺术",
                                "语言(非英语)", "旅游资讯", "综合预定", "民航", "铁路", "酒店", "行程管理", "民宿短租", "出国", "工具", "亲子儿童", "母婴",
                                "驾校", "违章", "汽车咨询", "汽车交易", "日常养车", "行车辅助", "租房", "买房", "装修家居", "电子产品", "问诊挂号", "养生保健",
                                "医疗服务", "减肥瘦身", "美妆美业", "菜谱", "餐饮店", "体育咨讯", "运动健身", "支付", "保险", "股票", "借贷", "理财", "彩票",
                                "记账", "银行", "美颜", "影像剪辑", "摄影修图", "相机", "绘画", "二手", "电商", "团购", "外卖", "电影票务", "社区超市",
                                "购物咨询", "笔记", "办公", "日程管理", "女性", "经营", "收款", "其他",
                                ]
            self.template = "这是一款[label]类软件"
            self.is_pre = True

        # label_num
        if (dataset_name in ['MNLI-m', 'MNLI-mm', 'SNLI']):
            self.label_num = 3
        elif (dataset_name in ['QNLI', 'RTE', 'MRPC', 'QQP']):
            self.label_num = 2




