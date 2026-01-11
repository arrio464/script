# README:
# Remote DNS lookup script
#
# Clash will rape your local DNS server. So you can't use local nslookup or dig to test DNS records.
# This script use remote API to perform nslookup.
#
# Usage:
#   python remote_nslookup.py example.com --dns-server cloudflare


### Author: Gemini 3 Pro
# prompt:
# write a remote nslookup script.
# 
# allow to choice dnsServer: cloudflare(default), google or authoritative
# display A, AAAA, CNAME and TXT records
# use urllib.request or other stand library intead of requests
# no interactive mode, to compatible with pipes
# API example:
# curl 'https://www.nslookup.io/api/v1/records' --data-raw '{"domain":"example.com","dnsServer":"cloudflare"}'


import argparse
import json
import sys
import urllib.request


def process_records(records, key, display_type):
    section = records.get(key, {})
    response = section.get("response", {})
    answers = response.get("answer", [])

    if not answers:
        return

    for item in answers:
        record = item.get("record", {})
        value = None

        if display_type == "A":
            value = record.get("ipv4")
        elif display_type == "AAAA":
            value = record.get("ipv6")
        elif display_type == "CNAME":
            value = record.get("target")
        elif display_type == "TXT":
            strings = record.get("strings")
            if strings:
                value = " ".join(f'"{s}"' for s in strings)

        if not value:
            value = record.get("raw")

        if value:
            print(f"{display_type:<6} {value}")


def main():
    parser = argparse.ArgumentParser(description="Remote DNS lookup tool")
    parser.add_argument("domain", help="Domain to lookup")
    parser.add_argument(
        "--dns-server",
        dest="dns_server",
        default="cloudflare",
        choices=["cloudflare", "google", "authoritative"],
        help="DNS server to use (default: cloudflare)",
    )

    args = parser.parse_args()

    url = "https://www.nslookup.io/api/v1/records"
    payload = {"domain": args.domain, "dnsServer": args.dns_server}

    try:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={
                "Content-Type": "application/json",
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            },
        )

        with urllib.request.urlopen(req) as response:
            if response.status != 200:
                print(
                    f"Error: API returned status code {response.status}",
                    file=sys.stderr,
                )
                sys.exit(1)

            body = response.read().decode("utf-8")
            data = json.loads(body)

            records = data.get("records", {})

            process_records(records, "a", "A")
            process_records(records, "aaaa", "AAAA")
            process_records(records, "cname", "CNAME")
            process_records(records, "txt", "TXT")

    except urllib.error.URLError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError:
        print("Error: Failed to parse JSON response", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
