# SSH Setup for GitHub

## Quick Setup (5 Steps)

### 1. Generate SSH Key
```bash
ssh-keygen -t ed25519 -C "your_email@example.com"
```

When prompted:
- **File location**: Press Enter (use default: `~/.ssh/id_ed25519`)
- **Passphrase**: Press Enter (no passphrase) or enter a secure passphrase

### 2. Start SSH Agent
```bash
eval "$(ssh-agent -s)"
```

Should show: `Agent pid [number]`

### 3. Add SSH Key to Agent
```bash
ssh-add ~/.ssh/id_ed25519
```

Should show: `Identity added: ~/.ssh/id_ed25519`

### 4. Copy Public Key
```bash
cat ~/.ssh/id_ed25519.pub
```

Copy the entire output (starts with `ssh-ed25519` and ends with your email)

### 5. Add Key to GitHub
1. Go to: **https://github.com/settings/keys**
2. Click **"New SSH key"**
3. **Title**: "MacBook Pro" (or any name)
4. **Key**: Paste the copied public key
5. Click **"Add SSH key"**

## Verify Connection

Test that SSH is working:
```bash
ssh -T git@github.com
```

Expected output:
```
Hi [username]! You've successfully authenticated, but GitHub does not provide shell access.
```

## Now Push Your Code

Once SSH is set up, run the commit script:
```bash
cd /Users/federicozahariev/Work/Programs/Richerme_Quantum_Hardware
chmod +x commit_and_push.sh
./commit_and_push.sh
```

The script will:
1. ✅ Automatically convert remote URL to SSH
2. ✅ Create commit with detailed message
3. ✅ Push to GitHub via SSH (no password needed!)

## Troubleshooting

### "Permission denied (publickey)"
- SSH key not added to GitHub
- Follow steps 4-5 above to add key

### "Could not open a connection to your authentication agent"
```bash
# Start SSH agent
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519
```

### Key file not found
```bash
# Check if key exists
ls -la ~/.ssh/

# If no id_ed25519, generate it:
ssh-keygen -t ed25519 -C "your_email@example.com"
```

### Multiple SSH keys
If you have multiple GitHub accounts:
```bash
# Generate key with custom name
ssh-keygen -t ed25519 -C "your_email@example.com" -f ~/.ssh/id_ed25519_work

# Add to SSH config
cat >> ~/.ssh/config <<EOF
Host github.com-work
    HostName github.com
    User git
    IdentityFile ~/.ssh/id_ed25519_work
EOF

# Use in git remote
git remote set-url origin git@github.com-work:fzahari/Richerme_hardwae_simulation.git
```

## macOS Keychain Integration

Save passphrase in macOS keychain (optional):
```bash
# Create/edit SSH config
cat >> ~/.ssh/config <<EOF
Host *
  AddKeysToAgent yes
  UseKeychain yes
  IdentityFile ~/.ssh/id_ed25519
EOF

# Add key with keychain
ssh-add --apple-use-keychain ~/.ssh/id_ed25519
```

## Alternative: HTTPS with Token

If you prefer HTTPS instead of SSH:
```bash
# Use HTTPS remote
git remote set-url origin https://github.com/fzahari/Richerme_hardwae_simulation.git

# You'll need a Personal Access Token:
# 1. Go to: https://github.com/settings/tokens
# 2. Generate new token (classic)
# 3. Select scope: repo
# 4. Use token as password when git asks
```

## Summary

**Recommended: SSH (no passwords needed)**
```bash
# One-time setup
ssh-keygen -t ed25519 -C "your_email@example.com"
ssh-add ~/.ssh/id_ed25519
cat ~/.ssh/id_ed25519.pub  # Add to GitHub

# Verify
ssh -T git@github.com

# Push (forever after)
./commit_and_push.sh  # Just works!
```

**Alternative: HTTPS (password/token each time)**
```bash
git remote set-url origin https://github.com/fzahari/Richerme_hardwae_simulation.git
git push  # Prompts for username + token
```

---

**Next step**: Once SSH is set up, run `./commit_and_push.sh` to push your code!
