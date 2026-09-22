import { assertDesktopProfileName } from "./profile-manager.js";
import { randomBytes } from "node:crypto";
import { closeSync, constants, fstatSync, lstatSync, openSync, readFileSync } from "node:fs";
import { dirname, isAbsolute, join } from "node:path";
import { PROFILE_TEMPLATES, resolveProfileDir } from "@deepseek-ai/dsh-app-boot";
import { Service } from "@deepseek-ai/cordis";
import { chmod, lstat, mkdir } from "node:fs/promises";
import { withFileLock, writeFileAtomic } from "@deepseek-ai/dsh-atomic-write";
//#region src/desktop-plugins.ts
/** Desktop-owned inventory and persistent disable state for direct profile bundles. */
const BIN_NAME = "dsh-plugin-desktop";
const STATE_VERSION = 1;
const STATE_FILE_MODE = 384;
const STATE_DIRECTORY_MODE = 448;
const MAX_STATE_BYTES = 64 * 1024;
const MAX_PROFILE_MANIFEST_BYTES = 1024 * 1024;
const MAX_PROFILES = 64;
const MAX_DISABLED_BUNDLES = 512;
const MAX_DIRECT_BUNDLES = 1024;
const PREVIEW_TTL_MS = 300 * 1e3;
const MAX_PREVIEWS = 256;
const BUNDLE_ID_PATTERN = /^bundle_[A-Za-z0-9_-]{32}$/u;
const DISABLE_PREVIEW_ID_PATTERN = /^disable_[A-Za-z0-9_-]{43}$/u;
const ENABLE_PREVIEW_ID_PATTERN = /^enable_[A-Za-z0-9_-]{43}$/u;
const PACKAGE_NAME_PATTERN = /^(?:@[a-z0-9][a-z0-9._-]*\/)?[a-z0-9][a-z0-9._-]*$/u;
const IMMUTABLE_BUNDLES = /* @__PURE__ */ new Set([
	...PROFILE_TEMPLATES.web ?? [],
	"@deepseek-ai/dsh-desktop-app",
	"dsh-plugin-desktop",
	"dsh-community-market"
]);
/** Error whose code is safe for a trusted Host integration to branch on. */
var DesktopPluginsError = class extends Error {
	code;
	constructor(code, message) {
		super(message);
		this.code = code;
		this.name = "DesktopPluginsError";
	}
};
function emptyState() {
	return {
		version: STATE_VERSION,
		profiles: []
	};
}
function safePackageName(value) {
	return typeof value === "string" && value.length <= 214 && PACKAGE_NAME_PATTERN.test(value);
}
function assertStateBootstrap(bootstrap) {
	assertDesktopProfileName(bootstrap.profileName);
	for (const [label, value] of [["Harness home", bootstrap.homeDir], ["state path", bootstrap.statePath]]) if (!isAbsolute(value) || value.includes("\0")) throw new Error(`${BIN_NAME}: desktop plugins ${label} must be an absolute path without NUL`);
}
function readProfileManifestBytes(path) {
	const directoryInfo = lstatSync(dirname(path));
	if (!directoryInfo.isDirectory() || directoryInfo.isSymbolicLink()) throw new Error(`${BIN_NAME}: active profile directory must be a real directory`);
	const pathInfo = lstatSync(path);
	if (!pathInfo.isFile() || pathInfo.isSymbolicLink()) throw new Error(`${BIN_NAME}: active profile manifest must be a regular file`);
	if (pathInfo.size > MAX_PROFILE_MANIFEST_BYTES) throw new Error(`${BIN_NAME}: active profile manifest is too large`);
	const descriptor = openSync(path, constants.O_RDONLY | (constants.O_NOFOLLOW ?? 0));
	try {
		const openedInfo = fstatSync(descriptor);
		if (!openedInfo.isFile() || openedInfo.size > MAX_PROFILE_MANIFEST_BYTES) throw new Error(openedInfo.size > MAX_PROFILE_MANIFEST_BYTES ? `${BIN_NAME}: active profile manifest is too large` : `${BIN_NAME}: active profile manifest must be a regular file`);
		const bytes = readFileSync(descriptor);
		if (bytes.byteLength > MAX_PROFILE_MANIFEST_BYTES) throw new Error(`${BIN_NAME}: active profile manifest is too large`);
		return bytes;
	} finally {
		closeSync(descriptor);
	}
}
function readDesktopProfileManifestInventory(bootstrap) {
	assertStateBootstrap(bootstrap);
	const bytes = readProfileManifestBytes(join(resolveProfileDir(bootstrap.profileName, bootstrap.homeDir), "package.json"));
	let parsed;
	try {
		parsed = JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(bytes));
	} catch (cause) {
		throw new Error(`${BIN_NAME}: invalid active profile manifest: ${cause instanceof Error ? cause.message : String(cause)}`);
	}
	if (parsed === null || typeof parsed !== "object" || Array.isArray(parsed)) throw new Error(`${BIN_NAME}: active profile manifest must hold a JSON object`);
	const root = parsed;
	const rawDependencies = root.dependencies;
	if (rawDependencies !== void 0 && (rawDependencies === null || typeof rawDependencies !== "object" || Array.isArray(rawDependencies))) throw new Error(`${BIN_NAME}: active profile manifest dependencies must be an object`);
	const dependencyNames = Object.keys(rawDependencies ?? {});
	if (dependencyNames.length > MAX_DIRECT_BUNDLES || dependencyNames.some((name) => !safePackageName(name))) throw new Error(`${BIN_NAME}: active profile manifest dependencies are invalid`);
	const dsh = root.dsh;
	if (dsh === void 0) return {
		bundleNames: [],
		dependencyNames: new Set(dependencyNames)
	};
	if (dsh === null || typeof dsh !== "object" || Array.isArray(dsh)) throw new Error(`${BIN_NAME}: active profile manifest dsh field must be an object`);
	const profile = dsh.profile;
	if (profile === void 0) return {
		bundleNames: [],
		dependencyNames: new Set(dependencyNames)
	};
	if (profile === null || typeof profile !== "object" || Array.isArray(profile)) throw new Error(`${BIN_NAME}: active profile manifest dsh.profile field must be an object`);
	const bundles = profile.bundles;
	if (bundles === void 0) return {
		bundleNames: [],
		dependencyNames: new Set(dependencyNames)
	};
	if (!Array.isArray(bundles) || bundles.length > MAX_DIRECT_BUNDLES || bundles.some((bundle) => !safePackageName(bundle))) throw new Error(`${BIN_NAME}: active profile manifest dsh.profile.bundles is invalid`);
	return {
		bundleNames: bundles,
		dependencyNames: new Set(dependencyNames)
	};
}
function readDesktopProfileBundleNames(bootstrap) {
	return readDesktopProfileManifestInventory(bootstrap).bundleNames;
}
function stableCompare(left, right) {
	return left < right ? -1 : left > right ? 1 : 0;
}
function parseState(value) {
	if (value === null || typeof value !== "object" || Array.isArray(value)) throw new Error("state root must be an object");
	const root = value;
	if (root.version !== STATE_VERSION || !Array.isArray(root.profiles)) throw new Error("state version or profiles list is invalid");
	if (root.profiles.length > MAX_PROFILES) throw new Error("state contains too many profiles");
	const profileNames = /* @__PURE__ */ new Set();
	const profiles = [];
	for (const rawProfile of root.profiles) {
		if (rawProfile === null || typeof rawProfile !== "object" || Array.isArray(rawProfile)) throw new Error("profile state must be an object");
		const profile = rawProfile;
		const profileName = profile.profileName;
		if (typeof profileName !== "string") throw new Error("profile state name is invalid");
		assertDesktopProfileName(profileName);
		if (profileNames.has(profileName)) throw new Error(`duplicate profile state ${JSON.stringify(profileName)}`);
		profileNames.add(profileName);
		const disabled = profile.disabledBundles;
		if (!Array.isArray(disabled) || disabled.length > MAX_DISABLED_BUNDLES || disabled.some((name) => !safePackageName(name))) throw new Error(`disabledBundles for profile ${JSON.stringify(profileName)} is invalid`);
		profiles.push({
			profileName,
			disabledBundles: [...new Set(disabled)].sort(stableCompare)
		});
	}
	profiles.sort((left, right) => stableCompare(left.profileName, right.profileName));
	return {
		version: STATE_VERSION,
		profiles
	};
}
function readState(statePath) {
	const directory = dirname(statePath);
	try {
		const directoryStat = lstatSync(directory);
		if (!directoryStat.isDirectory() || directoryStat.isSymbolicLink()) throw new Error(`${BIN_NAME}: plugin-management state directory is not private`);
	} catch (cause) {
		if (cause.code === "ENOENT") return emptyState();
		throw cause;
	}
	let stat;
	try {
		stat = lstatSync(statePath);
	} catch (cause) {
		if (cause.code === "ENOENT") return emptyState();
		throw cause;
	}
	if (!stat.isFile() || stat.isSymbolicLink()) throw new Error(`${BIN_NAME}: plugin-management state must be a regular file`);
	if (stat.size > MAX_STATE_BYTES) throw new Error(`${BIN_NAME}: plugin-management state is too large`);
	let parsed;
	try {
		const content = readFileSync(statePath, "utf8");
		if (Buffer.byteLength(content, "utf8") > MAX_STATE_BYTES) throw new Error("plugin-management state is too large");
		parsed = JSON.parse(content);
		return parseState(parsed);
	} catch (cause) {
		throw new Error(`${BIN_NAME}: invalid plugin-management state at ${statePath}: ${cause instanceof Error ? cause.message : String(cause)}`);
	}
}
/** Read disabled package names without changing Desktop-owned state. */
function readDesktopDisabledBundles(statePath, profileName) {
	assertDesktopProfileName(profileName);
	const profile = readState(statePath).profiles.find((candidate) => candidate.profileName === profileName);
	return new Set(profile?.disabledBundles ?? []);
}
/** Remove only the Desktop disable markers belonging to one deleted profile. */
async function clearDesktopProfilePluginState(statePath, profileName) {
	assertDesktopProfileName(profileName);
	if (!isAbsolute(statePath) || statePath.includes("\0")) throw new Error(`${BIN_NAME}: plugin-management state path must be absolute and contain no NUL`);
	await ensurePrivateStateDirectory(statePath);
	await withFileLock(statePath, async () => {
		const state = readState(statePath);
		if (!state.profiles.some((profile) => profile.profileName === profileName)) return;
		await writeFileAtomic(statePath, renderState(parseState({
			version: STATE_VERSION,
			profiles: state.profiles.filter((profile) => profile.profileName !== profileName)
		})), {
			mode: STATE_FILE_MODE,
			dirMode: STATE_DIRECTORY_MODE
		});
	});
}
/**
* Read the active profile's direct bundle declarations without resolving or
* parsing any bundle patch. This remains available when a bundle itself is
* what prevents the normal profile loader from starting.
*/
function readDesktopProfileBundleInventory(bootstrap) {
	const manifest = readDesktopProfileManifestInventory(bootstrap);
	const disabled = new Set(readDesktopDisabledBundles(bootstrap.statePath, bootstrap.profileName));
	const seen = /* @__PURE__ */ new Set();
	const bundles = [];
	for (const packageName of manifest.bundleNames) {
		if (seen.has(packageName)) continue;
		seen.add(packageName);
		const mutable = desktopPluginBundleMutable(packageName);
		bundles.push({
			packageName,
			status: mutable && disabled.has(packageName) ? "disabled" : "active",
			mutable,
			uninstallable: mutable && manifest.dependencyNames.has(packageName)
		});
	}
	return bundles;
}
/** Only explicit product bundles are immutable; every other resolved direct layer is user-disableable. */
function desktopPluginBundleMutable(packageName) {
	return safePackageName(packageName) && !IMMUTABLE_BUNDLES.has(packageName);
}
/** Filter only mutable layers named in Desktop-private disable state. */
function activeDesktopProfileLayers(profile, disabledBundles) {
	return profile.layers.filter((layer) => !(desktopPluginBundleMutable(layer.packageName) && disabledBundles.has(layer.packageName)));
}
async function ensurePrivateStateDirectory(statePath) {
	const directory = dirname(statePath);
	await mkdir(directory, {
		recursive: true,
		mode: STATE_DIRECTORY_MODE
	});
	const stat = await lstat(directory);
	if (!stat.isDirectory() || stat.isSymbolicLink()) throw new Error(`${BIN_NAME}: plugin-management state directory is not private`);
	await chmod(directory, STATE_DIRECTORY_MODE);
}
/**
* Persist one manifest-declared mutable bundle disable for the next
* generation. The caller's authorization callback runs while the state lock
* is held, immediately before the manifest and state are re-read.
*/
async function disableDesktopProfileBundle(bootstrap, packageName, authorize = () => {}) {
	let authorizationFailure;
	try {
		assertStateBootstrap(bootstrap);
		if (!safePackageName(packageName)) throw new DesktopPluginsError("invalid-target", "The Desktop plugin target is no longer available.");
		if (!desktopPluginBundleMutable(packageName)) throw new DesktopPluginsError("immutable-target", "This Desktop bundle cannot be disabled.");
		await ensurePrivateStateDirectory(bootstrap.statePath);
		await withFileLock(bootstrap.statePath, async () => {
			try {
				await authorize();
			} catch (cause) {
				authorizationFailure = cause;
				throw cause;
			}
			if (!readDesktopProfileBundleNames(bootstrap).includes(packageName) || !desktopPluginBundleMutable(packageName)) throw new DesktopPluginsError("invalid-target", "The Desktop plugin target is no longer available.");
			const state = readState(bootstrap.statePath);
			const existingProfile = state.profiles.find((candidate) => candidate.profileName === bootstrap.profileName);
			const disabled = new Set(existingProfile?.disabledBundles ?? []);
			if (disabled.has(packageName)) throw new DesktopPluginsError("already-disabled", "This Desktop bundle is already disabled.");
			disabled.add(packageName);
			const profiles = state.profiles.filter((candidate) => candidate.profileName !== bootstrap.profileName);
			if (existingProfile === void 0 && profiles.length >= MAX_PROFILES) throw new Error("plugin-management state contains too many profiles");
			if (disabled.size > MAX_DISABLED_BUNDLES) throw new Error("plugin-management state contains too many disabled bundles");
			profiles.push({
				profileName: bootstrap.profileName,
				disabledBundles: [...disabled].sort(stableCompare)
			});
			profiles.sort((left, right) => stableCompare(left.profileName, right.profileName));
			const rendered = renderState(parseState({
				version: STATE_VERSION,
				profiles
			}));
			if (Buffer.byteLength(rendered, "utf8") > MAX_STATE_BYTES) throw new Error("plugin-management state is too large");
			await writeFileAtomic(bootstrap.statePath, rendered, {
				mode: STATE_FILE_MODE,
				dirMode: STATE_DIRECTORY_MODE
			});
		});
		return { packageName };
	} catch (cause) {
		if (cause === authorizationFailure) throw cause;
		if (cause instanceof DesktopPluginsError) throw cause;
		throw new DesktopPluginsError("persistence-failed", "Unable to persist the Desktop plugin change.");
	}
}
/**
* Remove one mutable manifest-declared bundle's disable marker for the next
* generation. No profile manifest or package dependency is modified. The
* caller's authorization callback and all target checks run under the state
* lock so a disposed generation or concurrent state edit cannot be reused.
*/
async function enableDesktopProfileBundle(bootstrap, packageName, authorize = () => {}) {
	let authorizationFailure;
	try {
		assertStateBootstrap(bootstrap);
		if (!safePackageName(packageName)) throw new DesktopPluginsError("invalid-target", "The Desktop plugin target is no longer available.");
		if (!desktopPluginBundleMutable(packageName)) throw new DesktopPluginsError("immutable-target", "This Desktop bundle cannot be enabled.");
		await ensurePrivateStateDirectory(bootstrap.statePath);
		await withFileLock(bootstrap.statePath, async () => {
			try {
				await authorize();
			} catch (cause) {
				authorizationFailure = cause;
				throw cause;
			}
			if (!readDesktopProfileBundleNames(bootstrap).includes(packageName) || !desktopPluginBundleMutable(packageName)) throw new DesktopPluginsError("invalid-target", "The Desktop plugin target is no longer available.");
			const state = readState(bootstrap.statePath);
			const existingProfile = state.profiles.find((candidate) => candidate.profileName === bootstrap.profileName);
			const disabled = new Set(existingProfile?.disabledBundles ?? []);
			if (!disabled.delete(packageName)) throw new DesktopPluginsError("already-active", "This Desktop bundle is already active.");
			const profiles = state.profiles.filter((candidate) => candidate.profileName !== bootstrap.profileName);
			if (disabled.size > 0) profiles.push({
				profileName: bootstrap.profileName,
				disabledBundles: [...disabled].sort(stableCompare)
			});
			profiles.sort((left, right) => stableCompare(left.profileName, right.profileName));
			const rendered = renderState(parseState({
				version: STATE_VERSION,
				profiles
			}));
			if (Buffer.byteLength(rendered, "utf8") > MAX_STATE_BYTES) throw new Error("plugin-management state is too large");
			await writeFileAtomic(bootstrap.statePath, rendered, {
				mode: STATE_FILE_MODE,
				dirMode: STATE_DIRECTORY_MODE
			});
		});
		return { packageName };
	} catch (cause) {
		if (cause === authorizationFailure) throw cause;
		if (cause instanceof DesktopPluginsError) throw cause;
		throw new DesktopPluginsError("persistence-failed", "Unable to persist the Desktop plugin change.");
	}
}
function renderState(state) {
	return `${JSON.stringify(state, void 0, 2)}\n`;
}
function assertBootstrap(bootstrap) {
	assertDesktopProfileName(bootstrap.profileName);
	for (const [label, value] of [
		["Harness home", bootstrap.homeDir],
		["state path", bootstrap.statePath],
		["install anchor", bootstrap.installAnchor]
	]) if (!isAbsolute(value) || value.includes("\0")) throw new Error(`${BIN_NAME}: desktop plugins ${label} must be an absolute path without NUL`);
}
/** Generation-scoped direct bundle inventory with two-phase persistent state changes. */
var DesktopPluginsService = class extends Service {
	bootstrap;
	now;
	bundleIds = /* @__PURE__ */ new Map();
	previews = /* @__PURE__ */ new Map();
	disposed = false;
	operation;
	constructor(ctx, bootstrap) {
		assertBootstrap(bootstrap);
		super(ctx, "desktopPlugins");
		this.bootstrap = bootstrap;
		this.now = bootstrap.now ?? Date.now;
		ctx.effect(() => () => {
			this.disposed = true;
			this.previews.clear();
			this.bundleIds.clear();
		}, "dsh-plugin-desktop: desktop plugins lifetime");
	}
	list() {
		this.assertActive();
		return readDesktopProfileBundleInventory(this.bootstrap).map((item) => ({
			...item,
			bundleId: this.bundleId(item.packageName)
		}));
	}
	isDisabled(packageName) {
		this.assertActive();
		if (!safePackageName(packageName)) throw new DesktopPluginsError("invalid-target", "The Desktop plugin package name is invalid.");
		return this.disabledPackageNames().includes(packageName);
	}
	disabledPackageNames() {
		this.assertActive();
		return [...readDesktopDisabledBundles(this.bootstrap.statePath, this.bootstrap.profileName)].sort(stableCompare);
	}
	previewDisable(bundleId) {
		this.assertActive();
		if (!BUNDLE_ID_PATTERN.test(bundleId)) throw this.invalidTarget();
		const target = this.list().find((item) => item.bundleId === bundleId);
		if (target === void 0) throw this.invalidTarget();
		if (!target.mutable) throw new DesktopPluginsError("immutable-target", "This Desktop bundle cannot be disabled.");
		if (target.status === "disabled") throw new DesktopPluginsError("already-disabled", "This Desktop bundle is already disabled.");
		const preview = this.mintPreview("disable", target.packageName);
		return {
			previewId: preview.previewId,
			profileName: preview.profileName,
			packageName: preview.packageName,
			expiresAt: new Date(preview.expiresAt).toISOString()
		};
	}
	executeDisable(previewId) {
		try {
			this.assertActive();
			if (!DISABLE_PREVIEW_ID_PATTERN.test(previewId)) return Promise.reject(this.expiredPreview());
			if (this.operation !== void 0) return Promise.reject(new DesktopPluginsError("persistence-failed", "Another Desktop plugin change is already running."));
			const preview = this.previews.get(previewId);
			this.previews.delete(previewId);
			if (preview === void 0 || preview.expiresAt <= this.now() || preview.action !== "disable" || preview.profileName !== this.bootstrap.profileName) return Promise.reject(this.expiredPreview());
			const operation = this.persistDisable(preview);
			this.operation = operation;
			operation.then(() => {
				if (this.operation === operation) this.operation = void 0;
			}, () => {
				if (this.operation === operation) this.operation = void 0;
			});
			return operation;
		} catch (cause) {
			return Promise.reject(cause);
		}
	}
	previewEnable(bundleId) {
		this.assertActive();
		if (!BUNDLE_ID_PATTERN.test(bundleId)) throw this.invalidTarget();
		const target = this.list().find((item) => item.bundleId === bundleId);
		if (target === void 0) throw this.invalidTarget();
		if (!target.mutable) throw new DesktopPluginsError("immutable-target", "This Desktop bundle cannot be enabled.");
		if (target.status === "active") throw new DesktopPluginsError("already-active", "This Desktop bundle is already active.");
		const preview = this.mintPreview("enable", target.packageName, this.disabledStatePath(target.packageName));
		return {
			previewId: preview.previewId,
			profileName: preview.profileName,
			packageName: preview.packageName,
			expiresAt: new Date(preview.expiresAt).toISOString()
		};
	}
	executeEnable(previewId) {
		try {
			this.assertActive();
			if (!ENABLE_PREVIEW_ID_PATTERN.test(previewId)) return Promise.reject(this.expiredPreview());
			if (this.operation !== void 0) return Promise.reject(new DesktopPluginsError("persistence-failed", "Another Desktop plugin change is already running."));
			const preview = this.previews.get(previewId);
			this.previews.delete(previewId);
			if (preview === void 0 || preview.expiresAt <= this.now() || preview.action !== "enable" || preview.profileName !== this.bootstrap.profileName) return Promise.reject(this.expiredPreview());
			const operation = this.persistEnable(preview);
			this.operation = operation;
			operation.then(() => {
				if (this.operation === operation) this.operation = void 0;
			}, () => {
				if (this.operation === operation) this.operation = void 0;
			});
			return operation;
		} catch (cause) {
			return Promise.reject(cause);
		}
	}
	bundleId(packageName) {
		let id = this.bundleIds.get(packageName);
		if (id === void 0) {
			id = `bundle_${randomBytes(24).toString("base64url")}`;
			this.bundleIds.set(packageName, id);
		}
		return id;
	}
	mintPreview(action, packageName, statePath) {
		this.prunePreviews();
		if (this.previews.size >= MAX_PREVIEWS) {
			const oldest = this.previews.keys().next().value;
			if (oldest !== void 0) this.previews.delete(oldest);
		}
		const previewId = `${action}_${randomBytes(32).toString("base64url")}`;
		const preview = {
			previewId,
			action,
			profileName: this.bootstrap.profileName,
			packageName,
			...statePath === void 0 ? {} : { statePath },
			expiresAt: this.now() + PREVIEW_TTL_MS
		};
		this.previews.set(previewId, preview);
		return preview;
	}
	async persistDisable(preview) {
		try {
			const current = this.list().find((item) => item.packageName === preview.packageName);
			if (current === void 0) throw this.invalidTarget();
			if (!current.mutable) throw new DesktopPluginsError("immutable-target", "This Desktop bundle cannot be disabled.");
			if (current.status === "disabled") throw new DesktopPluginsError("already-disabled", "This Desktop bundle is already disabled.");
			return await disableDesktopProfileBundle(this.bootstrap, preview.packageName, () => {
				this.assertActive();
			});
		} catch (cause) {
			if (cause instanceof DesktopPluginsError) throw cause;
			throw new DesktopPluginsError("persistence-failed", "Unable to persist the Desktop plugin change.");
		}
	}
	async persistEnable(preview) {
		try {
			const current = this.list().find((item) => item.packageName === preview.packageName);
			if (current === void 0) throw this.invalidTarget();
			if (!current.mutable) throw new DesktopPluginsError("immutable-target", "This Desktop bundle cannot be enabled.");
			if (current.status === "active") throw new DesktopPluginsError("already-active", "This Desktop bundle is already active.");
			return await enableDesktopProfileBundle({
				...this.bootstrap,
				...preview.statePath === void 0 ? {} : { statePath: preview.statePath }
			}, preview.packageName, () => {
				this.assertActive();
			});
		} catch (cause) {
			if (cause instanceof DesktopPluginsError) throw cause;
			throw new DesktopPluginsError("persistence-failed", "Unable to persist the Desktop plugin change.");
		}
	}
	disabledStatePath(packageName) {
		if (readDesktopDisabledBundles(this.bootstrap.statePath, this.bootstrap.profileName).has(packageName)) return this.bootstrap.statePath;
	}
	prunePreviews() {
		const now = this.now();
		for (const [id, preview] of this.previews) if (preview.expiresAt <= now) this.previews.delete(id);
	}
	invalidTarget() {
		return new DesktopPluginsError("invalid-target", "The Desktop plugin target is no longer available.");
	}
	expiredPreview() {
		return new DesktopPluginsError("preview-expired", "The Desktop plugin confirmation expired or was already used.");
	}
	assertActive() {
		if (this.disposed) throw new Error(`${BIN_NAME}: desktopPlugins service disposed`);
	}
};
//#endregion
export { DesktopPluginsError, DesktopPluginsService, DesktopPluginsService as default, activeDesktopProfileLayers, clearDesktopProfilePluginState, desktopPluginBundleMutable, disableDesktopProfileBundle, enableDesktopProfileBundle, readDesktopDisabledBundles, readDesktopProfileBundleInventory };

//# sourceMappingURL=desktop-plugins.js.map