# St-Paywall Setup Guide

This guide will help you complete the setup of st-paywall for your Streamlit app.

## Overview

St-paywall requires TWO configurations:
1. **Authentication** - User login via Google OAuth (or another OIDC provider)
2. **Payment** - Subscription verification via Stripe or Buy Me A Coffee

## Step 1: Set Up Google OAuth Authentication

### 1.1 Create Google OAuth Credentials

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select an existing one
3. Enable the Google+ API:
   - Go to "APIs & Services" > "Library"
   - Search for "Google+ API"
   - Click "Enable"

4. Create OAuth 2.0 Credentials:
   - Go to "APIs & Services" > "Credentials"
   - Click "Create Credentials" > "OAuth 2.0 Client ID"
   - Select "Web application"
   - Add authorized redirect URIs:
     - For local testing: `http://localhost:8501/oauth2callback`
     - For production: `https://your-app-name.streamlit.app/oauth2callback`
   - Click "Create"

5. Copy your Client ID and Client Secret

### 1.2 Configure OAuth Consent Screen

1. Go to "APIs & Services" > "OAuth consent screen"
2. Choose "External" (unless you have a Google Workspace)
3. Fill in the required fields:
   - App name: "Your App Name"
   - User support email: your email
   - Developer contact: your email
4. Add scopes:
   - Click "Add or Remove Scopes"
   - Add: `openid`, `email`, `profile`
5. Save and continue

### 1.3 Update secrets.toml

Edit `.streamlit/secrets.toml` and update the `[auth]` section:

```toml
[auth]
redirect_uri = "http://localhost:8501/oauth2callback"  # Change for production
cookie_secret = "generate_a_random_32_char_string_here"  # Use a password generator
client_id = "YOUR_ACTUAL_CLIENT_ID.apps.googleusercontent.com"
client_secret = "YOUR_ACTUAL_CLIENT_SECRET"
server_metadata_url = "https://accounts.google.com/.well-known/openid-configuration"
```

**Important:** Generate a random string for `cookie_secret`. You can use Python:
```python
import secrets
print(secrets.token_urlsafe(32))
```

## Step 2: Configure Stripe Payment (Already Partially Done)

You've already added your Stripe test credentials. Here's what's configured:

✅ Test API Key
✅ Test Payment Link
⚠️ Production API Key (needs updating)
⚠️ Production Payment Link (needs updating)

### 2.1 Verify Test Mode Setup

Your current test configuration looks good:
- `stripe_api_key_test`: Already set
- `stripe_link_test`: Already set
- `testing_mode = true`: Enabled

### 2.2 Production Setup (When Ready)

When you're ready to go live:

1. Get your live Stripe API key from [Stripe Dashboard](https://dashboard.stripe.com/apikeys)
2. Create a live payment link at [Stripe Payment Links](https://dashboard.stripe.com/payment-links)
3. Update `.streamlit/secrets.toml`:
   ```toml
   testing_mode = false
   stripe_api_key = "sk_live_YOUR_ACTUAL_LIVE_KEY"
   stripe_link = "https://buy.stripe.com/YOUR_LIVE_LINK"
   ```

## Step 3: Test the Setup

### 3.1 Install Dependencies

```bash
pip install -r requirements.txt
```

### 3.2 Run the App Locally

```bash
streamlit run damage_streamlit.py
```

### 3.3 Expected Flow

1. **First visit**: You'll see "Login Required" with a Google login button
2. **After login**: You'll see a subscription prompt with Stripe payment link
3. **After subscription**: You'll see "✅ You have premium access!" and access to all tabs

### 3.4 Testing Stripe Subscriptions

Use Stripe test card numbers:
- Success: `4242 4242 4242 4242`
- Any future expiry date
- Any 3-digit CVC

More test cards: [Stripe Testing Docs](https://stripe.com/docs/testing)

## Step 4: Deploy to Streamlit Cloud (Optional)

### 4.1 Prepare for Deployment

1. Update Google OAuth redirect URI for your production URL:
   ```
   https://your-app-name.streamlit.app/oauth2callback
   ```

2. Update `.streamlit/secrets.toml` in Streamlit Cloud:
   - Go to your app settings
   - Paste your secrets in the "Secrets" section

### 4.2 Update secrets.toml for Production

```toml
[auth]
redirect_uri = "https://your-app-name.streamlit.app/oauth2callback"
# ... rest of auth config ...

testing_mode = false  # Use production Stripe
```

## Troubleshooting

### Error: "AttributeError: st.user has no attribute 'is_logged_in'"

This means authentication is not configured. Check:
- `[auth]` section exists in secrets.toml
- Google OAuth credentials are correct
- Redirect URI matches exactly

### Error: "Invalid client_id"

- Verify your Google OAuth Client ID is correct
- Make sure you're using the full ID including `.apps.googleusercontent.com`

### Subscription Not Detected

- Check Stripe API key is correct
- Verify the email used in Stripe matches the Google login email
- Check Stripe webhook settings if applicable

## Security Notes

1. **Never commit secrets.toml to version control**
   - It's already in .gitignore
   - Keep it local and in Streamlit Cloud secrets only

2. **Use environment-specific configurations**
   - Test mode for development
   - Production mode only when ready

3. **Rotate secrets periodically**
   - Change cookie_secret every few months
   - Update OAuth credentials if compromised

## Support Resources

- [Streamlit Authentication Docs](https://docs.streamlit.io/develop/concepts/connections/authentication)
- [st-paywall Documentation](https://st-paywall.readthedocs.io/)
- [Stripe Testing Guide](https://stripe.com/docs/testing)
- [Google OAuth Setup](https://developers.google.com/identity/protocols/oauth2)

## Next Steps

1. ✅ Complete Google OAuth setup
2. ✅ Test local login flow
3. ✅ Test subscription flow
4. ✅ Verify all tabs are accessible after subscription
5. 🚀 Deploy to production when ready
