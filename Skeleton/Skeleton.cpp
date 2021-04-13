//=============================================================================================
// Mintaprogram: Zöld háromszög. Ervenyes 2019. osztol.
//
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
// Nev    : 
// Neptun : 
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

enum MaterialType {ROUGH, REFLECTIVE};
struct Material {
	vec3 ka, kd, ks;
	float  shininess;
	vec3 F0;
	MaterialType type;
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
	ReflectiveMaterial(vec3 n, vec3 kappa) : Material(REFLECTIVE) {
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
		float t1 = (-b + sqrt_discr) / 2.0f / a;	// t1 >= t2 for sure
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
					hit.normal = plainNormal;
					hit.t = t;
					//vec3 n(0.17f, 0.35f, 1.5f), kappa(3.1f, 2.7f, 1.9f);
					vec3 n(0, 0, 0), kappa(1, 1, 1);
					hit.material = new ReflectiveMaterial(n, kappa);
					for (int n = 0; n < 5; n++) {
						vec3 side = f[i][n] - f[i][(n + 1) % 5];
						vec3 point = f[i][n] - hit.position;
						float cosAlpha = dot(normalize(side), normalize(point));
						//float alfa = acosf(cosAlpha);
						float dist = sqrt(1 - pow(cosAlpha, 2)) * length(point);
						//float dist = sinf(alfa) * length(hit.position);
						//printf("%lf", dist);
						if (dist < 0.1f) {
							hit.material = this->material;
							break;
						}
					}

					//hit.material = this->material;
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
	float a = 15;
	float b = 5;
	float c = 7;
	
	MiddleObject(Material* _material) {
		material = _material;
	}

	Hit intersect(const Ray& ray) {
		Hit hit;
		//vec3 dist = ray.start - center;
		float eA = a * pow(ray.dir.x, 2) + b * pow(ray.dir.y, 2);
		float eB = a * 2 * ray.start.x * ray.dir.x + b * 2 * ray.start.y * ray.dir.y - c * ray.dir.z;
		float eC = a * pow(ray.start.x, 2) + b * pow(ray.start.y, 2) - c * ray.start.z;

		float discr = pow(eB, 2) - 4 * eA * eC;

		if (discr < 0) return hit;
		float sqrt_discr = sqrtf(discr);
		float t1 = (-eB + sqrt_discr) / 2.0f / eA;	// t1 >= t2 for sure
		float t2 = (-eB - sqrt_discr) / 2.0f / eA;
		if (t1 <= 0) return hit;
		//hit.t = (t2 > 0) ? t2 : t1;
		vec3 t1Pos= ray.start + ray.dir * t1;
		vec3 t2Pos= ray.start + ray.dir * t2;
		if (length(t2Pos) <= sphereRadius && t2>0) {
			hit.position = t2Pos;
			hit.t = t2;			
		}
		else if (length(t1Pos) <= sphereRadius) {
			hit.position = t1Pos;
			hit.t = t1;
		}
		else return hit;

		//hit.position = ray.start + ray.dir * hit.t;
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

//float rnd() { return (float)rand() / RAND_MAX; }

const float epsilon = 0.0001f;

class Scene {
	std::vector<Intersectable*> objects;
	std::vector<Light*> lights;
	Camera camera;
	vec3 La;
public:
	void build() {
		vec3 eye = vec3(0, 0, 1), vup = vec3(0, 1, 0), lookat = vec3(0, 0, 0);
		//vec3 eye = vec3(0, 0, 10), vup = vec3(0, 10, 0), lookat = vec3(0, 0, 0);
		float fov = 60 * M_PI / 180;
		camera.set(eye, lookat, vup, fov);

		//La = vec3(0.4f, 0.4f, 0.4f);
		La = vec3(0.01f, 0.01f, 0.01f);
		//La = vec3(0.0f, 0.0f, 0.0f);
		//vec3 lightPoint(0.9, -0.9, 0.9), Le(2, 2, 2);
		vec3 lightPoint(1, -1, 1), Le(2, 2, 2);
		lights.push_back(new Light(lightPoint, Le));

		vec3 kd(0.2f, 0.3f, 1.5f), ks(3.1f, 2.7f, 1.9f);
		vec3 n(0.17f, 0.35f, 1.5f), kappa(3.1f, 2.7f, 1.9f);
		//Material* material = new Material(kd, ks, 100);
		RoughMaterial* materialRough = new RoughMaterial(kd, ks, 100);
		RoughMaterial* materialRough2 = new RoughMaterial(kd*2, ks, 100);
		ReflectiveMaterial* gold = new ReflectiveMaterial(n, kappa);
		//for (int i = 0; i < 500; i++)
			//objects.push_back(new Sphere(vec3( 0,  0,  0), 0.1f, gold));
			//objects.push_back(new Sphere(vec3( 0,  0,  0), 2, materialRough));
			//objects.push_back(new Sphere(vec3( 0.3,  0.3,  0.3), 0.1f, materialRough));
			objects.push_back(new MiddleObject(gold));
			//objects.push_back(new Sphere(vec3( -0.5,  0.5,  -0.5), 0.1f, materialRough2));
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
			Hit hit = object->intersect(ray); //  hit.t < 0 if no intersection
			if (hit.t > 0 && (bestHit.t < 0 || hit.t < bestHit.t))  bestHit = hit;
		}
		if (dot(ray.dir, bestHit.normal) > 0) bestHit.normal = bestHit.normal * (-1);
		return bestHit;
	}

	bool shadowIntersect(Ray ray) {	// for directional lights
		for (Intersectable* object : objects) if (object->intersect(ray).t > 0) return true;
		return false;
	}

	vec3 trace(Ray ray, int depth = 0) {
		if (depth > 5) return La;
		Hit hit = firstIntersect(ray);
		//printf("%lf %lf %lf\n", hit.position.x, hit.position.y, hit.position.z);
		if (hit.t < 0) return La;
		vec3 outRadiance = vec3(0, 0, 0);

		if (hit.material->type == ROUGH) {
			outRadiance = hit.material->ka * La;
			//vec3 outRadiance ;
			for (Light* light : lights) {
				vec3 direction = normalize(hit.position - light->point);
				//vec3 direction = normalize(light->point-hit.position  );
				double d = length(hit.position - light->point);
				Ray shadowRay(hit.position + -hit.normal * epsilon, direction);
				float cosTheta = dot(hit.normal, -direction);
				if (cosTheta > 0 && !shadowIntersect(shadowRay)) {	// shadow computation
					//outRadiance = light->Le/(d*d) * hit.material->kd * cosTheta;				
					outRadiance = outRadiance + light->Le / pow(d, 2) * hit.material->kd * cosTheta;
					vec3 halfway = normalize(-ray.dir + direction);
					float cosDelta = dot(hit.normal, halfway);
					if (cosDelta > 0) outRadiance = outRadiance + light->Le / pow(d, 2) * hit.material->ks * powf(cosDelta, hit.material->shininess);
				}
			}
		}
		if (hit.material->type == REFLECTIVE) {
			vec3 reflectedDir = ray.dir - hit.normal * dot(hit.normal, ray.dir) * 2.0f;
			float cosa = -dot(ray.dir, hit.normal);
			vec3 one(1, 1, 1);
			vec3 F= hit.material->F0 + (one - hit.material->F0) * pow(1 - cosa, 5);
			outRadiance = outRadiance + trace(Ray(hit.position + hit.normal * epsilon, reflectedDir), depth+1) * F;
		}
		
		return outRadiance;
	}

	void Animate(float dt) { camera.Animate(dt); }
};

GPUProgram gpuProgram; // vertex and fragment shaders
Scene scene;

// vertex shader in GLSL
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

// fragment shader in GLSL
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
	unsigned int vao;	// vertex array object id and texture id
	Texture texture;
public:
	FullScreenTexturedQuad(int windowWidth, int windowHeight, std::vector<vec4>& image)
		: texture(windowWidth, windowHeight, image)
	{
		glGenVertexArrays(1, &vao);	// create 1 vertex array object
		glBindVertexArray(vao);		// make it active

		unsigned int vbo;		// vertex buffer objects
		glGenBuffers(1, &vbo);	// Generate 1 vertex buffer objects

		// vertex coordinates: vbo0 -> Attrib Array 0 -> vertexPosition of the vertex shader
		glBindBuffer(GL_ARRAY_BUFFER, vbo); // make it active, it is an array
		float vertexCoords[] = { -1, -1,  1, -1,  1, 1,  -1, 1 };	// two triangles forming a quad
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertexCoords), vertexCoords, GL_STATIC_DRAW);	   // copy to that part of the memory which is not modified 
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);     // stride and offset: it is tightly packed

		//glGenTextures(1, &textureId);
		//glBindTexture(GL_TEXTURE_2D, textureId);
		//glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		//glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	}

	void LoadTexture(std::vector<vec4>& image) {
		texture.create(windowWidth, windowHeight, image);
	}

	void Draw() {
		glBindVertexArray(vao);	// make the vao and its vbos active playing the role of the data source
		gpuProgram.setUniform(texture, "textureUnit");
		glDrawArrays(GL_TRIANGLE_FAN, 0, 4);	// draw two triangles forming a quad
	}
};

FullScreenTexturedQuad* fullScreenTexturedQuad;

// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	scene.build();

	std::vector<vec4> image(windowWidth * windowHeight);
	long timeStart = glutGet(GLUT_ELAPSED_TIME);
	scene.render(image);
	long timeEnd = glutGet(GLUT_ELAPSED_TIME);
	printf("Rendering time: %d milliseconds\n", (timeEnd - timeStart));

	// copy image to GPU as a texture
	fullScreenTexturedQuad = new FullScreenTexturedQuad(windowWidth, windowHeight, image);

	// create program for the GPU
	gpuProgram.create(vertexSource, fragmentSource, "fragmentColor");
}

// Window has become invalid: Redraw
void onDisplay() {
	std::vector<vec4> image(windowWidth * windowHeight);
	long timeStart = glutGet(GLUT_ELAPSED_TIME);
	scene.render(image);
	long timeEnd = glutGet(GLUT_ELAPSED_TIME);
	printf("Rendering time: %d milliseconds\n", (timeEnd - timeStart));

	// copy image to GPU as a texture
	fullScreenTexturedQuad->LoadTexture(image);
	fullScreenTexturedQuad->Draw();
	glutSwapBuffers();									// exchange the two buffers
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {

}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) {
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
	scene.Animate(0.1f);
	glutPostRedisplay();
}
