import os
import discord
import asyncio
import schedule
import time
from googletrans import Translator
from discord import app_commands
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import praw

# Load environment variables
load_dotenv()
TOKEN = os.getenv('DISCORD_TOKEN')

# Hàm chuyển đổi an toàn từ string sang int (xử lý cả định dạng khoa học)
def safe_convert_to_int(value):
    if value is None:
        return None
    try:
        # Nếu có ký tự 'E' (định dạng khoa học) thì chuyển qua float trước
        if 'E' in value or 'e' in value:
            return int(float(value))
        return int(value)
    except (ValueError, TypeError) as e:
        print(f"Error converting value: {value}. Error: {e}")
        return None

# Lấy và chuyển đổi các ID
CHANNEL_ID = safe_convert_to_int(os.getenv('DISCORD_CHANNEL_ID'))
ADMIN_USER_ID = safe_convert_to_int(os.getenv('ADMIN_USER_ID'))

# In ra để debug
print(f"CHANNEL_ID: {CHANNEL_ID}")
print(f"ADMIN_USER_ID: {ADMIN_USER_ID}")

# Kiểm tra các biến bắt buộc
if not TOKEN:
    print("DISCORD_TOKEN is missing. Exiting.")
    exit(1)

if not CHANNEL_ID:
    print("DISCORD_CHANNEL_ID is missing or invalid. Exiting.")
    exit(1)

if not ADMIN_USER_ID:
    print("ADMIN_USER_ID is missing or invalid. Exiting.")
    exit(1)

# Initialize Discord client
intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)
tree = app_commands.CommandTree(client)

# Initialize Reddit client
reddit = praw.Reddit(
    client_id=os.getenv('REDDIT_CLIENT_ID'),
    client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
    user_agent="MicroTaskScraper/1.0"
)

# Initialize translator
translator = Translator()

# Subreddits and keywords
SUBREDDITS = ["slavelabour", "forhire", "Jobs4Bitcoins", "taskrabbit"]
KEYWORDS = ["task", "micro job", "hiring", "help needed", "small job"]

# Hashtag classifier setup
hashtag_data = {
    "data": [
        "design graphic design logo",
        "writing content blog article",
        "programming python javascript",
        "translation english vietnamese",
        "data entry excel spreadsheet"
    ],
    "target": ["#Design", "#Content", "#Tech", "#Translation", "#DataEntry"]
}
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(hashtag_data['data'])
clf = MultinomialNB().fit(X, hashtag_data['target'])

# Database simulation
processed_posts = set()
pending_approvals = {}

def predict_hashtag(text):
    try:
        text_vector = vectorizer.transform([text])
        return clf.predict(text_vector)[0]
    except Exception as e:
        print(f"Hashtag prediction error: {e}")
        return "#General"

def translate_content(text):
    try:
        return translator.translate(text, src='en', dest='vi').text
    except Exception as e:
        print(f"Translation error: {e}")
        return text

async def fetch_new_posts():
    await client.wait_until_ready()
    channel = client.get_channel(CHANNEL_ID)
    if not channel:
        print(f"Error: Cannot find channel with ID {CHANNEL_ID}")
        return
    
    print(f"Starting to fetch new posts at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    for sub in SUBREDDITS:
        print(f"Checking subreddit: r/{sub}")
        try:
            subreddit = reddit.subreddit(sub)
            for post in subreddit.new(limit=10):
                if post.id in processed_posts:
                    continue
                
                processed_posts.add(post.id)
                
                title_lower = post.title.lower()
                if not any(kw in title_lower for kw in KEYWORDS):
                    continue
                
                # Translate content
                print(f"New post found: {post.title[:50]}...")
                translated_title = translate_content(post.title)
                translated_content = translate_content(post.selftext[:500]) if post.selftext else ""
                hashtag = predict_hashtag(post.title)
                original_url = f"https://reddit.com{post.permalink}"
                
                # Save for approval
                pending_approvals[post.id] = {
                    "title": translated_title,
                    "content": translated_content,
                    "hashtag": hashtag,
                    "original_url": original_url
                }
                
                # Send to admin for approval
                try:
                    user = await client.fetch_user(ADMIN_USER_ID)
                    embed = discord.Embed(
                        title=f"📝 Bài viết mới cần hiệu đính: {translated_title[:200]}",
                        description=translated_content[:2000],
                        color=0x3498db
                    )
                    embed.add_field(name="Hashtag dự kiến", value=hashtag, inline=False)
                    embed.add_field(name="Duyệt bài", value=f"✅ `/approve {post.id}`\n❌ `/reject {post.id}`", inline=False)
                    embed.add_field(name="Link gốc", value=original_url, inline=False)
                    await user.send(embed=embed)
                    print(f"Sent approval request for post: {post.id}")
                except Exception as e:
                    print(f"Error sending DM to admin: {e}")
                
        except Exception as e:
            print(f"Error fetching from r/{sub}: {e}")

@tree.command(name="approve", description="Duyệt bài viết")
async def approve_post(interaction, post_id: str):
    if interaction.user.id != ADMIN_USER_ID:
        return await interaction.response.send_message("❌ Bạn không có quyền thực hiện thao tác này!", ephemeral=True)
    
    if post_id not in pending_approvals:
        return await interaction.response.send_message("❌ Bài viết không tồn tại hoặc đã hết hạn!", ephemeral=True)
    
    post = pending_approvals[post_id]
    channel = client.get_channel(CHANNEL_ID)
    if not channel:
        return await interaction.response.send_message("❌ Không tìm thấy kênh đích!", ephemeral=True)
    
    try:
        embed = discord.Embed(
            title=post["title"][:200],
            description=post["content"][:2000],
            color=0x2ecc71
        )
        embed.add_field(name="Hashtag", value=post["hashtag"], inline=False)
        embed.add_field(name="Nguồn", value=post["original_url"], inline=False)
        embed.set_footer(text="✅ Đã được phê duyệt")
        
        await channel.send(embed=embed)
        del pending_approvals[post_id]
        await interaction.response.send_message(f"✅ Đã đăng bài viết {post_id}!")
    except Exception as e:
        await interaction.response.send_message(f"❌ Lỗi khi đăng bài: {str(e)[:100]}")

@tree.command(name="reject", description="Từ chối bài viết")
async def reject_post(interaction, post_id: str):
    if interaction.user.id != ADMIN_USER_ID:
        return await interaction.response.send_message("❌ Bạn không có quyền thực hiện thao tác này!", ephemeral=True)
    
    if post_id in pending_approvals:
        del pending_approvals[post_id]
    await interaction.response.send_message(f"❌ Đã từ chối bài viết {post_id}!")

@client.event
async def on_ready():
    await tree.sync()
    print(f"Logged in as {client.user.name} (ID: {client.user.id})")
    print(f"Target channel ID: {CHANNEL_ID}")
    print(f"Admin user ID: {ADMIN_USER_ID}")
    
    # Start scheduled tasks
    schedule.every(10).minutes.do(lambda: asyncio.create_task(fetch_new_posts()))
    
    # Chạy ngay lần đầu
    await fetch_new_posts()
    
    # Lặp chạy schedule
    while True:
        schedule.run_pending()
        await asyncio.sleep(60)

if __name__ == "__main__":
    try:
        client.run(TOKEN)
    except discord.errors.LoginFailure:
        print("❌ Lỗi đăng nhập Discord. Vui lòng kiểm tra TOKEN.")
    except Exception as e:
        print(f"❌ Lỗi không xác định: {e}")
