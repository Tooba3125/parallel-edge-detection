# Small (400x300)
wget -q -O test_images/small.jpg "https://picsum.photos/id/100/400/300"
if [ -f test_images/small.jpg ] && [ $(stat -c%s test_images/small.jpg 2>/dev/null || echo "1000") -gt 1000 ]; then
    echo "small.jpg (400x300)"
else
    echo "Download failed, using fallback"
fi

# Medium (1080x720)
wget -q -O test_images/medium.jpg "https://picsum.photos/id/101/1080/720"
echo "medium.jpg (1080x720)"

# Large (1920x1080)
wget -q -O test_images/large.jpg "https://picsum.photos/id/104/1920/1080"
echo "large.jpg (1920x1080)"

echo ""
echo "Converting JPG to PNG..."
for img in test_images/*.jpg; do
    if [ -f "$img" ]; then
        convert "$img" "${img%.jpg}.png" 2>/dev/null
        if [ -f "${img%.jpg}.png" ]; then
            rm "$img"
            echo "  Converted $(basename ${img%.jpg}.png)"
        fi
    fi
done


