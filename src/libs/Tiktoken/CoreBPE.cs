using System.Collections;
using System.Collections.Concurrent;
using System.Text.RegularExpressions;
using System.Text.Unicode;
using Tiktoken.Utilities;

namespace Tiktoken;

/// <summary>
/// 
/// </summary>
public class CoreBpe
{
    private IReadOnlyDictionary<string, int> SpecialTokensEncoder { get; set; }
    private IReadOnlyDictionary<byte[], int> Encoder { get; set; }
    private IReadOnlyDictionary<string, int> FastEncoder { get; set; }

    internal bool EnableCache { get; set; } = true;
    private IDictionary<string, IReadOnlyCollection<int>> FastCache { get; set; } =
        new ConcurrentDictionary<string, IReadOnlyCollection<int>>();
    private IDictionary<string, int> FastCacheCounts { get; set; } =
        new ConcurrentDictionary<string, int>();

    private Regex SpecialRegex { get; set; }
    private Regex Regex { get; set; }

    private IReadOnlyDictionary<int, byte[]> Decoder { get; set; }
    private IReadOnlyDictionary<int, string> SpecialTokensDecoder { get; set; }

    /// <summary>
    /// 
    /// </summary>
    /// <param name="encoder"></param>
    /// <param name="specialTokensEncoder"></param>
    /// <param name="pattern"></param>
    public CoreBpe(
        IReadOnlyDictionary<byte[], int> encoder,
        IReadOnlyDictionary<string, int> specialTokensEncoder,
        string pattern)
    {
        encoder = encoder ?? throw new ArgumentNullException(nameof(encoder));
        specialTokensEncoder = specialTokensEncoder ?? throw new ArgumentNullException(nameof(specialTokensEncoder));
        pattern = pattern ?? throw new ArgumentNullException(nameof(pattern));
        
        Encoder = encoder;
        FastEncoder = Encoder
            .ToDictionary(
                static x => new string(x.Key.Select(y => (char) y).ToArray()),
                static x => x.Value);
        SpecialTokensEncoder = specialTokensEncoder;
        
        Regex = new Regex(pattern, RegexOptions.Compiled);
        SpecialRegex = new Regex("(" + string.Join("|", specialTokensEncoder.Keys.Select(Regex.Escape)) + ")", RegexOptions.Compiled);

        Decoder = Encoder
            .ToDictionary(
                static x => x.Value,
                static x => x.Key);
        SpecialTokensDecoder = specialTokensEncoder
            .ToDictionary(
                static x => x.Value,
                static x => x.Key);
    }

    /// <summary>
    /// 
    /// </summary>
    /// <param name="text"></param>
    /// <returns></returns>
    public int CountTokensNative(string text)
    {
        text = text ?? throw new ArgumentNullException(nameof(text));
        
        var tokens = 0;
        var textSpan = text.AsSpan();

        foreach (var match in Regex.EnumerateMatches(textSpan))
        {
            var matchValue = textSpan.Slice(match.Index, match.Length).ToArray();
            var fastKey = new string(textSpan.Slice(match.Index, match.Length));

            if (FastEncoder.ContainsKey(fastKey))
            {
                tokens++;
                continue;
            }
            if (EnableCache && FastCacheCounts.TryGetValue(fastKey, out var fastNumberOfTokens))
            {
                tokens += fastNumberOfTokens;
                continue;
            }

            var piece = System.Text.Encoding.UTF8.GetBytes(matchValue);
            if (Encoder.ContainsKey(piece))
            {
                tokens++;
                continue;
            }
            
            var numberOfTokens = BytePairEncoding.BytePairEncodeCountTokens(piece, Encoder);
            tokens += numberOfTokens;

            if (EnableCache)
            {
                FastCacheCounts[fastKey] = numberOfTokens;
            }
        }

        return tokens;
    }
    
    /// <summary>
    /// 
    /// </summary>
    /// <param name="text"></param>
    /// <param name="allowedSpecial"></param>
    /// <param name="disallowedSpecial"></param>
    /// <returns></returns>
    public IReadOnlyCollection<int> EncodeNative(
        string text,
        HashSet<string> allowedSpecial,
        HashSet<string> disallowedSpecial)
    {
        text = text ?? throw new ArgumentNullException(nameof(text));
        allowedSpecial = allowedSpecial ?? throw new ArgumentNullException(nameof(allowedSpecial));
        disallowedSpecial = disallowedSpecial ?? throw new ArgumentNullException(nameof(disallowedSpecial));
        
        var tokens = new List<int>();
        var textSpan = text.AsSpan();

        var specialTokens = new List<(int Index, int Length)>(capacity: 32);

        foreach (var match in SpecialRegex.EnumerateMatches(textSpan))
        {
            var value = textSpan.Slice(start: match.Index, length: match.Length).ToString();

            if (disallowedSpecial.Contains(value))
            {
                throw new InvalidOperationException(value);
            }
            if (allowedSpecial.Contains(value))
            {
                specialTokens.Add((match.Index, match.Length));
            }
            else
            {
                throw new InvalidOperationException("Invalid special token sets");
            }
        }
        specialTokens.Add((Index: text.Length, Length: 0));

        var start = 0;
        foreach (var (specialStart, specialLength) in specialTokens)
        {
            foreach (var match in Regex.EnumerateMatches(textSpan[start..specialStart]))
            {
                var matchValue = textSpan.Slice(match.Index, match.Length).ToArray();
                var fastKey = new string(textSpan.Slice(match.Index, match.Length));

                if (FastEncoder.TryGetValue(fastKey, out var fastToken))
                {
                    tokens.Add(fastToken);
                    continue;
                }
                if (EnableCache && FastCache.TryGetValue(fastKey, out var fastTokens))
                {
                    tokens.AddRange(fastTokens);
                    continue;
                }
                
                var piece = System.Text.Encoding.UTF8.GetBytes(matchValue);
                if (Encoder.TryGetValue(piece, out var token))
                {
                    tokens.Add(token);
                    continue;
                }
                
                var pair = BytePairEncoding.BytePairEncode(piece, Encoder);
                tokens.AddRange(pair);

                if (EnableCache)
                {
                    FastCache[fastKey] = pair;
                }
            }

            if (specialLength != 0)
            {
                start = specialStart + specialLength;
                var piece = new string(textSpan.Slice(specialStart, specialLength));

                var token = SpecialTokensEncoder[piece];
                tokens.Add(token);
            }
        }

        return tokens;
    }
    
    /// <summary>
    /// 
    /// </summary>
    /// <param name="text"></param>
    /// <param name="allowedSpecial"></param>
    /// <param name="disallowedSpecial"></param>
    /// <returns></returns>
    public IReadOnlyCollection<string> Explore(
        string text,
        HashSet<string> allowedSpecial,
        HashSet<string> disallowedSpecial)
    {
        text = text ?? throw new ArgumentNullException(nameof(text));
        allowedSpecial = allowedSpecial ?? throw new ArgumentNullException(nameof(allowedSpecial));
        disallowedSpecial = disallowedSpecial ?? throw new ArgumentNullException(nameof(disallowedSpecial));

        static bool GetBit(byte b, int bitNumber) {
            return (b & (1 << bitNumber)) != 0;
        }
        
        var values = new List<string>();
        var textSpan = text.AsSpan();

        var specialTokens = new List<(int Index, int Length)>(capacity: 32);

        foreach (var match in SpecialRegex.EnumerateMatches(textSpan))
        {
            var value = textSpan.Slice(start: match.Index, length: match.Length).ToString();
            if (disallowedSpecial.Contains(value))
            {
                throw new InvalidOperationException(value);
            }
            if (allowedSpecial.Contains(value))
            {
                specialTokens.Add((match.Index, match.Length));
            }
            else
            {
                throw new InvalidOperationException("Invalid special token sets");
            }
        }
        specialTokens.Add((Index: text.Length, Length: 0));

        var start = 0;
        foreach (var (specialStart, specialLength) in specialTokens)
        {
            foreach (var match in Regex.EnumerateMatches(textSpan[start..specialStart]))
            {
                var matchValue = textSpan.Slice(match.Index, match.Length).ToArray();
                var fastKey = new string(textSpan.Slice(match.Index, match.Length));


                var piece = System.Text.Encoding.UTF8.GetBytes(matchValue);
                if (Encoder.ContainsKey(piece))
                {
                    values.Add(fastKey);
                    continue;
                }
                
                var pair = BytePairEncoding.BytePairExplore(piece, Encoder);
                foreach (var bytes in pair)
                {
                    var value = System.Text.Encoding.UTF8.GetString(bytes);
                    
                    values.Add(value);
                }
            }

            if (specialLength != 0)
            {
                start = specialStart + specialLength;
                var piece = new string(textSpan.Slice(specialStart, specialLength));

                values.Add(piece);
            }
        }

        return values;
    }
    
    public IReadOnlyCollection<Tuple<string, int>> ExploreUtfSafe(
        string text,
        HashSet<string> allowedSpecial,
        HashSet<string> disallowedSpecial)
    {
        text = text ?? throw new ArgumentNullException(nameof(text));
        allowedSpecial = allowedSpecial ?? throw new ArgumentNullException(nameof(allowedSpecial));
        disallowedSpecial = disallowedSpecial ?? throw new ArgumentNullException(nameof(disallowedSpecial));
        
        var values = new List<Tuple<string, int>>();
        var textSpan = text.AsSpan();

        var specialTokens = new List<(int Index, int Length)>(capacity: 32);

        foreach (var match in SpecialRegex.EnumerateMatches(textSpan))
        {
            var value = textSpan.Slice(start: match.Index, length: match.Length).ToString();
            if (disallowedSpecial.Contains(value))
            {
                throw new InvalidOperationException(value);
            }
            if (allowedSpecial.Contains(value))
            {
                specialTokens.Add((match.Index, match.Length));
            }
            else
            {
                throw new InvalidOperationException("Invalid special token sets");
            }
        }
        specialTokens.Add((Index: text.Length, Length: 0));

        List<byte> accumulatedBytes = [];
        
        var start = 0;
        foreach (var (specialStart, specialLength) in specialTokens)
        {
            foreach (var match in Regex.EnumerateMatches(textSpan[start..specialStart]))
            {
                var matchValue = textSpan.Slice(match.Index, match.Length).ToArray();
                var fastKey = new string(textSpan.Slice(match.Index, match.Length));


                var piece = System.Text.Encoding.UTF8.GetBytes(matchValue);
                if (Encoder.ContainsKey(piece))
                {
                    values.Add(new Tuple<string, int>(fastKey, 1));
                    continue;
                }
                
                var pair = BytePairEncoding.BytePairExplore(piece, Encoder);
                
                accumulatedBytes.Clear();
                int accuCount = 1;

                for (int i = 0; i < pair.Count; i++)
                {
                    var currentPair = pair[i];
                    
                    accumulatedBytes.AddRange(currentPair);
                    byte[] accuArr = [.. accumulatedBytes];
                    
                    if (Utf8.IsValid(accuArr))
                    {
                        var value = System.Text.Encoding.UTF8.GetString(accuArr);
                        values.Add(new Tuple<string, int>(value, accuCount));  
                        accumulatedBytes.Clear();
                        accuCount = 1;
                    }
                    else
                    {
                        accuCount++;
                    }
                }

                if (accumulatedBytes.Count > 0)
                {
                    var value = System.Text.Encoding.UTF8.GetString([.. accumulatedBytes]);
                    values.Add(new Tuple<string, int>(value, accuCount));  
                    accumulatedBytes.Clear();
                }
            }

            if (specialLength != 0)
            {
                start = specialStart + specialLength;
                var piece = new string(textSpan.Slice(specialStart, specialLength));

                values.Add(new Tuple<string, int>(piece, 1));
            }
        }

        return values;
    }

    /// <summary>
    /// 
    /// </summary>
    /// <param name="tokens"></param>
    /// <returns></returns>
    public byte[] DecodeNative(IReadOnlyCollection<int> tokens)
    {
        tokens = tokens ?? throw new ArgumentNullException(nameof(tokens));
        
        var ret = new List<byte>(tokens.Count * 2);
        foreach (var token in tokens)
        {
            byte[] tokenBytes = Array.Empty<byte>();
            if (Decoder.TryGetValue(token, out var value))
            {
                tokenBytes = value;
            } 
            else
            {
                if (SpecialTokensDecoder.TryGetValue(token, out var valueS))
                {
                    tokenBytes = System.Text.Encoding.UTF8.GetBytes(valueS);
                }
            }

            if (tokenBytes.Length > 0)
            {
                ret.AddRange(tokenBytes);
            } 
        }
        return ret.ToArray();
    }
}