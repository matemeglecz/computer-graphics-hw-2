//=============================================================================================
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat, BOM kihuzando.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kiveve
// - Mashonnan atvett programresszleteket forrasmegjeloles nelkul felhasznalni es
// - felesleges programsorokat a beadott programban hagyni!!!!!!! 
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak 
// A keretben nem szereplo GLUT fuggvenyek tiltottak.
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : Meglecz Mate	
// Neptun : A7RBKU
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================
#include "framework.h"

enum MaterialType {ROUGH, REFLECTIVE, REFLECTIVE_ROTATE};
struct Material {
	vec3 ka, kd, ks;
	float  shininess;
	vec3 F0;
	MaterialType type;
	vec3 middle;
	Material(MaterialType t) { type=t; }
};

struct RoughMaterial : Material {
	RoughMaterial(vec3 _kd, vec3 _ks, float _shininess) : Material(ROUGH) {
		ka = (_kd * M_PI);
		kd = (_kd);
		ks = (_ks);
		shininess = _shininess;
	}
};

vec3 operator/(vec3 num, vec3 denom) {
	return vec3(num.x / denom.x, num.y / denom.y, num.z / denom.z);
}
struct  ReflectiveMaterial:Material{
	ReflectiveMaterial(vec3 n, vec3 kappa, MaterialType type=REFLECTIVE) : Material(type) {
		vec3 one(1, 1, 1);
		F0 = ((n - one) * (n - one) + kappa * kappa) / ((n + one) * (n + one) + kappa * kappa);
	}
};

struct Hit {
	float t;
	vec3 position, normal;
	Material* material;
	Hit() { t = -1; }
};

struct Ray {
	vec3 start, dir;
	Ray(vec3 _start, vec3 _dir) { start = _start; dir = normalize(_dir); }
};

class Intersectable {
protected:
	Material* material;
public:
	virtual Hit intersect(const Ray& ray) = 0;
};

struct Sphere : public Intersectable {
	vec3 center;
	float radius;

	Sphere(const vec3& _center, float _radius, Material* _material) {
		center = _center;
		radius = _radius;
		material = _material;
	}

	Hit intersect(const Ray& ray) {
		Hit hit;
		vec3 dist = ray.start - center;
		float a = dot(ray.dir, ray.dir);
		float b = dot(dist, ray.dir) * 2.0f;
		float c = dot(dist, dist) - radius * radius;
		float discr = b * b - 4.0f * a * c;
		if (discr < 0) return hit;
		float sqrt_discr = sqrtf(discr);
		float t1 = (-b + sqrt_discr) / 2.0f / a;	
		float t2 = (-b - sqrt_discr) / 2.0f / a;
		if (t1 <= 0) return hit;
		hit.t = (t2 > 0) ? t2 : t1;
		hit.position = ray.start + ray.dir * hit.t;
		hit.normal = (hit.position - center) * (1.0f / radius);
		hit.material = material;
		return hit;
	}
};

struct Dodekaedron : public Intersectable {
	vec3 v[20] = { vec3(0, 0.618, 1.618), vec3(0, -0.618, 1.618), vec3(0, -0.618, -1.618), vec3(0, 0.618, -1.618),
						vec3(1.618, 0, 0.618), vec3(-1.618, 0, 0.618), vec3(-1.618, 0, -0.618), vec3(1.618, 0, -0.618),
						vec3(0.618, 1.618, 0), vec3(-0.618, 1.618, 0), vec3(-0.618, -1.618, 0), vec3(0.618, -1.618, 0),
						vec3(1, 1, 1), vec3(-1, 1, 1), vec3(-1, -1, 1), vec3(1, -1, 1),
						vec3(1, -1, -1), vec3(1, 1, -1), vec3(-1, 1, -1), vec3(-1, -1, -1) };

	vec3 f[12][5] = { {v[0], v[1], v[15], v[4], v[12]}, {v[0], v[12], v[8], v[9], v[13]}, {v[0], v[13], v[5], v[14], v[1]}, {v[1], v[14], v[10], v[11], v[15]},
					{v[2], v[3], v[17], v[7], v[16]}, {v[2], v[16], v[11], v[10], v[19]}, {v[2], v[19], v[6], v[18], v[3]}, {v[18], v[9], v[8], v[17], v[3]},
					{v[15], v[11], v[16], v[7], v[4]}, {v[4], v[7], v[17], v[8], v[12]}, {v[13], v[9], v[18], v[6], v[5]}, {v[5], v[6], v[19], v[10], v[14]}
					}; 
	Material* mirror = new ReflectiveMaterial(vec3(0, 0, 0), vec3(1, 1, 1), REFLECTIVE_ROTATE);

	Dodekaedron(Material* _material) {
		material = _material;
	}

	Hit intersect(const Ray& ray) {
		Hit hit;
	
		for (int i = 0; i < 12; i++) {
			vec3 plainNormal = normalize(cross(f[i][1] - f[i][0], f[i][2] - f[i][0]));
			float t = dot((f[i][0] - ray.start), plainNormal) / dot(ray.dir, plainNormal);
			if (t <= 0) continue;
			vec3 hitPoint = ray.start + ray.dir * t;
			bool inside = true;
			vec3 middlePoint = vec3(0, 0, 0);
			for (int l = 0; l < 5; l++) {
				middlePoint = middlePoint + f[i][l];
			}
			middlePoint = middlePoint / 5;
			for (int j = 0; j < 5; j++) {
				vec3 triangleV[3] = { f[i][j] , f[i][(j + 1) % 5] ,  middlePoint};
				for (int k = 0; k < 3 && inside; k++) {
					plainNormal = cross(triangleV[(1+k)%3] - triangleV[0+k], triangleV[(2+k)%3] - triangleV[0+k]);
					if (!(dot(cross(triangleV[(1 + k) % 3] - triangleV[0 + k], hitPoint - triangleV[0 + k]), plainNormal) > 0)) inside = false;
				}
				if (inside && (hit.t>t || hit.t<0)) {
					hit.position = hitPoint;
					hit.normal = normalize(plainNormal);
					hit.t = t;				
					hit.material = mirror;
					for (int n = 0; n < 5; n++) {
						vec3 side = f[i][n] - f[i][(n + 1) % 5];
						vec3 point = f[i][n] - hit.position;
						float cosAlpha = dot(normalize(side), normalize(point));
						float dist = sqrt(1 - pow(cosAlpha, 2)) * length(point);
						if (dist < 0.1f) {
							hit.material = this->material;
							break;
						}
					}
					hit.material->middle = middlePoint;
					break;
				}
				inside = true;
			}
		}
		return hit;
	}
};

struct MiddleObject : public Intersectable {
	float sphereRadius = 0.3f;
	vec3 center = vec3(0, 0, 0);
	float a = 6.1f;
	float b = 6.1f;
	float c = 1.1f;
	
	MiddleObject(Material* _material) {
		material = _material;
	}

	Hit intersect(const Ray& ray) {
		Hit hit;

		float eA = a * pow(ray.dir.x, 2) + b * pow(ray.dir.y, 2);
		float eB = a * 2.0f * ray.start.x * ray.dir.x + b * 2.0f * ray.start.y * ray.dir.y - c * ray.dir.z;
		float eC = a * pow(ray.start.x, 2) + b * pow(ray.start.y, 2) - c * ray.start.z;

		float discr = pow(eB, 2) - 4 * eA * eC;

		if (discr < 0) return hit;
		float sqrt_discr = sqrtf(discr);
		float t1 = (-eB + sqrt_discr) / 2.0f / eA;	
		float t2 = (-eB - sqrt_discr) / 2.0f / eA;
		if (t1 <= 0) return hit;
		vec3 t1Pos= ray.start + ray.dir * t1;
		vec3 t2Pos= ray.start + ray.dir * t2;
		if (length(t2Pos) <= sphereRadius ) { 
			hit.position = t2Pos;
			hit.t = t2;			
		}
		else if (length(t1Pos) <= sphereRadius) {
			hit.position = t1Pos;
			hit.t = t1;
		}
		else return hit;

		vec3 xDer = vec3(1, 0, (2 * a * hit.position.x) / c);
		vec3 yDer = vec3(0, 1, (2 * b * hit.position.y) / c);
		hit.normal = cross(xDer, yDer);
		hit.normal = normalize(hit.normal);

		hit.material = material;
		return hit;
	}

};

class Camera {
	vec3 eye, lookat, right, up;
	float fov;
public:
	void set(vec3 _eye, vec3 _lookat, vec3 vup, float _fov) {
		eye = _eye;
		lookat = _lookat;
		fov = _fov;
		vec3 w = eye - lookat;
		float focus = length(w);
		right = normalize(cross(vup, w)) * focus * tanf(fov / 2);
		up = normalize(cross(w, right)) * focus * tanf(fov / 2);
	}
	Ray getRay(int X, int Y) {
		vec3 dir = lookat + right * (2.0f * (X + 0.5f) / windowWidth - 1) + up * (2.0f * (Y + 0.5f) / windowHeight - 1) - eye;
		return Ray(eye, dir);
	}

	void Animate(float dt) {
		vec3 d = eye - lookat;
		eye = vec3(d.x * cos(dt) + d.z * sin(dt), d.y, -d.x * sin(dt) + d.z * cos(dt)) + lookat;
		set(eye, lookat, up, fov);
	}
};

struct Light {
	vec3 point;
	vec3 Le;
	Light(vec3 _point, vec3 _Le) {
		point = _point;
		Le = _Le;
	}
};

const float epsilon = 0.0001f;

class Scene {
	std::vector<Intersectable*> objects;
	std::vector<Light*> lights;
	Camera camera;
	vec3 La;
public:
	void build() {
		vec3 eye = vec3(0, 0, 1.3), vup = vec3(0, 1, 0), lookat = vec3(0, 0, 0);
		float fov = 45 * M_PI / 180;
		camera.set(eye, lookat, vup, fov);

		La = vec3(0.1f, 0.1f, 0.1f);
		vec3 lightPoint(1, -1, 1), Le(2, 2, 2);
		lights.push_back(new Light(lightPoint, Le));

		vec3 kd(0.2f, 1.3f, 1.5f), ks(3.1f, 2.7f, 1.9f);
		vec3 n(0.17f, 0.35f, 1.5f), kappa(3.1f, 2.7f, 1.9f);
		RoughMaterial* materialRough = new RoughMaterial(kd, ks, 100);
		ReflectiveMaterial* gold = new ReflectiveMaterial(n, kappa, REFLECTIVE);
		objects.push_back(new MiddleObject(gold));
		objects.push_back(new Dodekaedron(materialRough));
	}

	void render(std::vector<vec4>& image) {
		for (int Y = 0; Y < windowHeight; Y++) {
#pragma omp parallel for
			for (int X = 0; X < windowWidth; X++) {
				vec3 color = trace(camera.getRay(X, Y));
				image[Y * windowWidth + X] = vec4(color.x, color.y, color.z, 1);
			}
		}
	}

	Hit firstIntersect(Ray ray) {
		Hit bestHit;
		for (Intersectable* object : objects) {
			Hit hit = object->intersect(ray); 
			if (hit.t > 0 && (bestHit.t < 0 || hit.t < bestHit.t))  bestHit = hit;
		}
		if (dot(ray.dir, bestHit.normal) > 0) bestHit.normal = bestHit.normal * (-1);
		return bestHit;
	}

	bool shadowIntersect(Ray ray, float t) {
		Hit hit = firstIntersect(ray);
		if (hit.t < t ) return false;
		return true;
	}

	vec3 trace(Ray ray, int depth = 0) {
		if (depth > 5) return La;
		Hit hit = firstIntersect(ray);
		if (hit.t < 0) return La;
		vec3 outRadiance = vec3(0, 0, 0);

		if (hit.material->type == ROUGH) {
			outRadiance = hit.material->ka * La;
			for (Light* light : lights) {
				vec3 direction = normalize(light->point-hit.position);
				double d = length(light->point- hit.position);
				Ray shadowRay(hit.position + hit.normal * epsilon, direction);
				float cosTheta = dot(hit.normal, direction);
				if (cosTheta > 0 && !shadowIntersect(shadowRay, d)) {				
					outRadiance = outRadiance + light->Le / pow(d, 2) * hit.material->kd * cosTheta;
					vec3 halfway = normalize(-ray.dir + direction);
					float cosDelta = dot(hit.normal, halfway);
					if (cosDelta > 0) outRadiance = outRadiance + light->Le / pow(d, 2) * hit.material->ks * powf(cosDelta, hit.material->shininess);
				}
			}
		}
		if (hit.material->type == REFLECTIVE || hit.material->type == REFLECTIVE_ROTATE) {
			vec3 reflectedDir = ray.dir - hit.normal * dot(hit.normal, ray.dir) * 2.0f;
			float cosa = -dot(ray.dir, hit.normal);
			vec3 one(1, 1, 1);
			vec3 F= hit.material->F0 + (one - hit.material->F0) * pow(1 - cosa, 5);
			if (hit.material->type == REFLECTIVE_ROTATE) {
				hit.position = hit.position- hit.material->middle;
				float theta = 2 * M_PI / 5;
				hit.position = hit.position * cos(theta) + cross(hit.position, hit.normal) * sin(theta) + hit.normal * dot(hit.normal, hit.position) * (1 - cos(theta));
				hit.position = hit.position + hit.material->middle;
				reflectedDir= reflectedDir * cos(theta) + cross(reflectedDir, hit.normal) * sin(theta) + hit.normal * dot(hit.normal, reflectedDir) * (1 - cos(theta));
			}

			outRadiance = outRadiance + trace(Ray(hit.position + hit.normal * epsilon, reflectedDir), depth+1) * F;
		}
		
		return outRadiance;
	}

	void Animate(float dt) { camera.Animate(dt); }
};

GPUProgram gpuProgram; 
Scene scene;

const char* vertexSource = R"(
	#version 330
    precision highp float;

	layout(location = 0) in vec2 cVertexPosition;	// Attrib Array 0
	out vec2 texcoord;

	void main() {
		texcoord = (cVertexPosition + vec2(1, 1))/2;							// -1,1 to 0,1
		gl_Position = vec4(cVertexPosition.x, cVertexPosition.y, 0, 1); 		// transform to clipping space
	}
)";

const char* fragmentSource = R"(
	#version 330
    precision highp float;

	uniform sampler2D textureUnit;
	in  vec2 texcoord;			// interpolated texture coordinates
	out vec4 fragmentColor;		// output that goes to the raster memory as told by glBindFragDataLocation

	void main() {
		fragmentColor = texture(textureUnit, texcoord); 
	}
)";

class FullScreenTexturedQuad {
	unsigned int vao;	
	Texture texture;
public:
	FullScreenTexturedQuad(int windowWidth, int windowHeight, std::vector<vec4>& image)
		: texture(windowWidth, windowHeight, image)
	{
		glGenVertexArrays(1, &vao);	
		glBindVertexArray(vao);		

		unsigned int vbo;		
		glGenBuffers(1, &vbo);	

		glBindBuffer(GL_ARRAY_BUFFER, vbo); 
		float vertexCoords[] = { -1, -1,  1, -1,  1, 1,  -1, 1 };	
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertexCoords), vertexCoords, GL_STATIC_DRAW);	    
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);    
	}

	void LoadTexture(std::vector<vec4>& image) {
		texture.create(windowWidth, windowHeight, image);
	}

	void Draw() {
		glBindVertexArray(vao);	
		gpuProgram.setUniform(texture, "textureUnit");
		glDrawArrays(GL_TRIANGLE_FAN, 0, 4);	
	}
};

FullScreenTexturedQuad* fullScreenTexturedQuad;

void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	scene.build();

	std::vector<vec4> image(windowWidth * windowHeight);
	long timeStart = glutGet(GLUT_ELAPSED_TIME);
	scene.render(image);
	long timeEnd = glutGet(GLUT_ELAPSED_TIME);
	printf("Rendering time: %d milliseconds\n", (timeEnd - timeStart));

	fullScreenTexturedQuad = new FullScreenTexturedQuad(windowWidth, windowHeight, image);

	gpuProgram.create(vertexSource, fragmentSource, "fragmentColor");
}

void onDisplay() {
	std::vector<vec4> image(windowWidth * windowHeight);
	long timeStart = glutGet(GLUT_ELAPSED_TIME);
	scene.render(image);
	long timeEnd = glutGet(GLUT_ELAPSED_TIME);
	printf("Rendering time: %d milliseconds\n", (timeEnd - timeStart));

	fullScreenTexturedQuad->LoadTexture(image);
	fullScreenTexturedQuad->Draw();
	glutSwapBuffers();									
}

void onKeyboard(unsigned char key, int pX, int pY) {
}

void onKeyboardUp(unsigned char key, int pX, int pY) {

}

void onMouse(int button, int state, int pX, int pY) {
}

void onMouseMotion(int pX, int pY) {
}

void onIdle() {
	scene.Animate(0.1f);
	glutPostRedisplay();
}
