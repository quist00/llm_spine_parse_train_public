```markdown
# Campus vLLM Service Exposure Plan

## Objective
Expose the campus vLLM endpoint on a non-localhost interface with strong protections:
- API key authentication
- TLS encryption
- Network-level access controls

This is a planning document only. No implementation changes are included in this plan.

## Scope Boundaries
Included:
- Protecting inference endpoints with API keys
- Adding TLS termination
- Restricting exposure to hardened entrypoints
- Validation, rollback, and operations guidance

Excluded:
- Model training changes
- LoRA quality tuning decisions
- Changes to the main non-campus Docker workflow

## Current State Summary
- vLLM is launched by `tools/start_vllm.py` through the campus wrapper `tools/start_vllm_campus.py`.
- Campus compose runtime is managed in `docker_campus/docker-compose.campus.yml`.
- Environment defaults are in `docker_campus/.env.campus`.
- vLLM is currently exposed through host port mapping and does not enforce API authentication.
- Jupyter and vLLM are separate services in the same compose stack.

## Target Architecture
Desired flow:  
Client -> HTTPS reverse proxy -> API key validation -> internal vLLM service

Design goals:
1. Keep raw vLLM backend on internal Docker network only.
2. Expose only one hardened public endpoint.
3. Reject unauthorized requests before they reach vLLM.
4. Add basic request controls and audit visibility.

## Phased Implementation Plan

### Phase 1: Authentication and Proxy Layer
1. Add a reverse proxy service to `docker_campus/docker-compose.campus.yml`.
2. Bind external inference traffic to the proxy service only.
3. Route proxy upstream to internal `vllm-campus` service.
4. Enforce API key requirement in proxy or auth sidecar.
5. Return `401` for missing or invalid credentials.

Success criteria:
1. Requests without `Authorization` fail with `401`.
2. Valid key requests succeed end-to-end.
3. vLLM remains reachable internally for proxy only.

### Phase 2: TLS
1. Add certificate and private key mount points for the proxy service.
2. Terminate TLS at the proxy.
3. Force HTTPS for external clients.
4. Keep internal service-to-service traffic on internal network.

Success criteria:
1. External clients can connect over HTTPS.
2. Certificates are presented correctly.
3. Plain external HTTP is disabled or redirected.

### Phase 3: Network Hardening
1. Remove direct external host exposure of raw vLLM backend port in `docker_campus/docker-compose.campus.yml`.
2. Restrict inbound firewall or security group rules to approved campus ranges.
3. Optionally add IP allowlist logic at proxy layer.

Success criteria:
1. Raw vLLM backend port is not externally reachable.
2. Only approved source ranges can access hardened endpoint.

### Phase 4: Operations and Reliability
1. Add request rate limits and body-size limits at proxy.
2. Log auth failures and request metadata.
3. Define key rotation workflow and emergency access procedure.
4. Add health checks for proxy and backend readiness.

Success criteria:
1. Service remains stable under expected load.
2. Security events are visible in logs.
3. Operational runbook supports rotation and recovery.

## Files To Update Later

Existing files:
1. `docker_campus/docker-compose.campus.yml`
   - Add proxy and auth path
   - Remove direct raw backend exposure
   - Add cert and config mounts

2. `docker_campus/.env.campus`
   - Add API key and TLS settings
   - Add optional allowlist and rate-limit settings

3. `tools/start_vllm.py`
   - Add optional startup wiring for auth-related settings if needed
   - Preserve backward compatibility when auth is not enabled

4. `tools/start_vllm_campus.py`
   - Add safe passthrough for auth and proxy mode environment toggles

5. `docker_campus/README.md`
   - Add secure endpoint setup
   - Add client auth usage examples
   - Add key rotation and rollback guidance

New files to create later:
1. Proxy configuration file in `docker_campus`
2. Optional auth-sidecar configuration in `docker_campus`
3. Optional service-operations runbook in `docker_campus`

## Client-Side Contract Changes
After rollout, clients should:
1. Use secure HTTPS base URL for inference endpoint.
2. Send `Authorization` header using Bearer token.
3. Continue calling OpenAI-compatible endpoints:
   - `/v1/models`
   - `/v1/completions`
   - `/v1/chat/completions`

## Validation Checklist

Functional:
1. Unauthenticated request returns `401`.
2. Invalid key returns `401`.
3. Valid key returns `200` on models endpoint.
4. Valid key completion works for base model and adapter model.

Security:
1. Raw vLLM backend port is not externally reachable.
2. TLS certificate is correctly served.
3. Firewall or security policy restricts source ranges as intended.

Operational:
1. Compose restart works cleanly.
2. Health checks are green for proxy and backend.
3. Logs show auth failures and response status trends.

## Rollback Plan
If hardening rollout causes issues:
1. Revert compose changes to previous working state.
2. Temporarily re-enable prior direct port mapping if needed for service recovery.
3. Restart affected service first, then full stack if required.
4. Validate models endpoint availability.
5. Re-apply hardening in smaller increments with checkpoints.

## Operations Notes

API key rotation:
1. Generate new key.
2. Add new key in environment or secret store.
3. Reload or redeploy proxy/auth layer.
4. Migrate clients.
5. Retire previous key.

Monitoring priorities:
1. `401` spikes
2. `429` spikes
3. Latency regressions
4. Backend unreachable errors

Recommended policy:
1. Keep Jupyter access tunnel-only.
2. Expose only hardened inference endpoint.
3. Never commit real API keys to git.
```

